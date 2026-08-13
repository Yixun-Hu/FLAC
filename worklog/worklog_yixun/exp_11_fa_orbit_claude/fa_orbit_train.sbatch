#!/bin/bash
# ============================================================================
# exp_11 ARM TRAINING LAUNCHER — one orbit arm at the ONE pinned P0 rung.
#
# Round-3 review (fa_orbit_codex_code_r3_review.md) rebuilt this script: the
# recipe is now literally pinned rather than operator-supplied, restarts get
# exp_10's torch.load depth, the run directory is locked, the exit taxonomy is a
# tested function, and the environment/W&B destinations are gated.
#
# RECIPE (post-P0, Yixun 2026-08-05): every arm runs WITH ViT gradient
# checkpointing. P0 measured the no-ckpt recipe INFEASIBLE for C8 and richer
# orbits (OOM at micro-8, 45,457 MiB) while checkpointed peaks at ~9.4 GB, so all
# four arms share one checkpointed rung and the ONLY delta between them is the
# averaging orbit.
#
# SUBMIT: never by hand — `fa_orbit_submit.sh <ARM>` derives every Slurm flag
# from the pins below, so an operator cannot mis-enter --gres/--mem/--time.
#   ARM                C4L | C8 | C16 | C32
#   EXPECT_SHA         full 40-hex reviewed commit OID (required). Binding is
#                      by CONTENT of the training surfaces, not HEAD identity:
#                      a launch is accepted when HEAD == EXPECT_SHA, or when
#                      the training closure is byte-identical between the two
#                      (two writers commit to this checkout; worklog/record
#                      commits must not kill a queued leg).
#   RESUME_CKPT/EXPECTED_STEP   crash restart only (see LINEAGE)
#   CHUNK_END          optional, RESTART only: stop THIS job at that step (a
#                      multiple of the 2500 checkpoint cadence, <= the budget) and
#                      let the next job resume from it. The budget pin does not
#                      move; only where this leg stops does. Chunk legs are
#                      chained by fa_orbit_chunk_watchdog.sh via the submitter.
#   SMOKE=1            the reviewed multi-GPU smoke (see SMOKE MODE)
# RUNG / MAXSTEPS / MIN_FREE_MB / time limit are NOT operator inputs any more.
#
# LINEAGE (fail-closed, exactly two stories):
#   INITIAL  no RESUME_CKPT, EXPECTED_STEP unset/0, run directory absent.
#   RESTART  EXPECTED_STEP > 0 AND RESUME_CKPT inside this arm's OWN
#            <RUNDIR>/checkpoints/ AND the checkpoint passes
#            fa_orbit_ckpt_preflight.py (embedded step/config/optimizer/
#            scheduler/EMA + binding to the original launch manifest).
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
# (FLAC_exp11_smoke_<ARM> / exp11_smoke_<ARM>) and its own save-dir prefix, so a
# smoke can never touch or resume an arm's real lineage.
#
# TEST HOOK: OUTPUT_ROOT (default outputs_FLAC) relocates the output namespace so
# the guard tests never write under a production prefix. It changes no gate.
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_train_%x_%j.out
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
PINNED_RUNG="8x8"                          # P0 run 1334933 + spot 9bf1936: fastest uniform rung where ALL arms fit (C32 peak 30,817 MiB)
PINNED_MB="8"                              # micro-batch per GPU (8 x 8 = 64 = eff = BN batch)
PINNED_NGPU="8"                            # ranks
PINNED_MAXSTEPS=100000                     # Q10: extended budget (was 40000, the
                                           # plan §2 primary matched step, which
                                           # remains the TABLE step — the extension
                                           # adds trajectory, it does not move the
                                           # registered comparison point)
PINNED_CHECKPOINT_EVERY=2500               # exp_07 cadence
PINNED_MIN_FREE_MB="36500"                 # batched C32 peak 32,063 MiB + ~4.4 GB margin (max-across-arms floor)
PINNED_TIME_LIMIT_C4L="24:00:00"           # batched 40k/0.6598 = 16.8 h x1.3 + startup
PINNED_TIME_LIMIT_C8="35:00:00"            # batched 40k/0.4351 = 25.5 h x1.3 + startup
PINNED_TIME_LIMIT_C16="60:00:00"           # batched 40k/0.2454 = 45.3 h x1.3 + startup
PINNED_TIME_LIMIT_C32="112:00:00"          # batched 40k/0.1308 = 84.9 h x1.3 + startup — SINGLE segment (no wall-split needed)
# VANL is the vanilla-conditioning arm of the SAME recipe (Q9): its cost comes
# from the official P0 VAN_8x8 rate, not from an orbit slope, because it makes no
# orbit passes at all — 40k/1.07 steps/s = 10.4 h x1.3 + startup.
PINNED_TIME_LIMIT_VANL="14:00:00"
# Q10 RESTART legs: 40k -> 100k is 60,000 further steps at the batched rates,
# x1.3 + startup. Each must sit under the 168 h partition cap, and each does.
PINNED_TIME_LIMIT_RESTART_C4L="34:00:00"    # 60k/0.6598 = 25.3 h
PINNED_TIME_LIMIT_RESTART_C8="51:00:00"     # 60k/0.4351 = 38.3 h
PINNED_TIME_LIMIT_RESTART_C16="89:00:00"    # 60k/0.2454 = 67.9 h
PINNED_TIME_LIMIT_RESTART_C32="160:00:00"   # 60k/0.1308 = 127.4 h (cap 168 h)
PINNED_TIME_LIMIT_RESTART_VANL="19:00:00"   # 60k/1.0722 = 15.5 h
# Round 5 CHUNKED legs: the partition never backfills a 34-160 h allocation, so a
# leg trains to the next 2500-step boundary and exits. One chunk at the batched
# rates is 2500/rate, rounded up with the same x1.3 margin plus a generous
# startup allowance (cold start + torch/PL import + the 40k-checkpoint load are a
# fixed cost per leg and now amortise over 2500 steps, not 60000):
#   C4L 2500/0.6598 = 1.05 h | C8 1.60 h | C16 2.83 h | C32 5.31 h | VANL 0.65 h
PINNED_TIME_LIMIT_CHUNK_C4L="03:00:00"
PINNED_TIME_LIMIT_CHUNK_C8="04:00:00"
PINNED_TIME_LIMIT_CHUNK_C16="06:00:00"
PINNED_TIME_LIMIT_CHUNK_C32="09:00:00"
PINNED_TIME_LIMIT_CHUNK_VANL="02:30:00"
PINNED_P0_MANIFEST_SHA256="72607b922177208d56055d604b292d697b643ef3b7ab48261ab2e23a0cc2b53b"  # batched matrix manifest bd96575-…-a3ed28eb; spot manifest sha in the commit message
# Environment pins (round-3 B6) — measured on the reviewed environment:
PINNED_PYTHON="/n/fs/gatrdp/envs/flac/bin/python"
PINNED_PL_VERSION="2.1.0"
PINNED_TORCH_VERSION="2.7.0+cu126"
PINNED_VAE_SHA256="8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9"
# ============================================================================

REPO=/n/fs/gatrdp/codespace/FLAC
# TEST HOOK (guard tests only): sbatch copies this script to a spool dir, so the
# repo path must be absolute; FA_ORBIT_REPO_OVERRIDE lets the guard suite point a
# dry run at a worktree. It is honoured ONLY outside a Slurm job and scrubbed
# immediately, so it can never influence a real launch.
if [ -n "${FA_ORBIT_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  REPO="$FA_ORBIT_REPO_OVERRIDE"
fi
unset FA_ORBIT_REPO_OVERRIDE
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
EXP07="$REPO/worklog/worklog_yixun/exp_07_fa_scratch_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

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
# CHUNK_END (round 5): the step THIS job stops at. Empty = the leg runs to the
# budget, which is every pre-round-5 leg. It narrows train.py's --max-steps only;
# PINNED_MAXSTEPS remains the campaign budget and its gate is untouched.
CHUNK_END="${CHUNK_END:-}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

die() { echo "$1"; exit "${2:-2}"; }

# --- A. parameters ------------------------------------------------------------
[ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32|VANL) - abort"
[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
case "$ARM" in
  C4L|C8|C16|C32|VANL) ;;
  *) die "ARM '${ARM}' is not a legal exp_11 arm — C4L|C8|C16|C32 only (FA1/VAN/CKPT4 are P0 profiling cells, never arms) - abort" ;;
esac
case "$EXPECTED_STEP" in ''|*[!0-9]*) die "EXPECTED_STEP '${EXPECTED_STEP}' must be a non-negative integer - abort";; esac

# --- B. the pins decide the recipe (round-3 B1) -------------------------------
if [ "$SMOKE" = "1" ]; then
  RUNG="${SMOKE_RUNG:-}"; MAXSTEPS="${SMOKE_MAXSTEPS:-30}"; MIN_FREE_MB="${SMOKE_MIN_FREE_MB:-}"
  CHECKPOINT_EVERY="${SMOKE_CHECKPOINT_EVERY:-10}"
  [ -n "$RUNG" ] || die "SMOKE=1 requires SMOKE_RUNG (32x2|16x4|8x8) - abort"
  [ -n "$MIN_FREE_MB" ] || die "SMOKE=1 requires SMOKE_MIN_FREE_MB (per-GPU floor) - abort"
  TIME_LIMIT="${SMOKE_TIME:-00:30:00}"; TIME_PIN_NAME="SMOKE_TIME"
  NAME="FLAC_exp11_smoke_${ARM}"; EXPNAME="exp11_smoke_${ARM}"
  SAVEDIR="${OUTPUT_ROOT}/exp11_smoke/${ARM}"
  echo "=== SMOKE MODE: pins bypassed, EVERY other gate active; identity ${EXPNAME} ==="
else
  # Q10 / re-pin fix 1: the wall pin follows the LEG, not the arm. A restart leg
  # is 60,000 further steps, not 40,000 from scratch, so the submitter allocates
  # PINNED_TIME_LIMIT_RESTART_<ARM>. The job selected PINNED_TIME_LIMIT_<ARM>
  # regardless and then rejected its own (correct) allocation in gate H — the
  # third hard-abort path the re-pin review found on jobs 3662828-30. The JOB now
  # selects the same pin the submitter did and enforces THAT one.
  # Round 5: a CHUNK leg is shorter again — it stops at the next boundary — so it
  # carries the arm's CHUNK pin. Submitter and job select by the same rule, and
  # gate H still enforces whichever pin was selected here.
  if [ "$EXPECTED_STEP" -gt 0 ] && [ -n "$CHUNK_END" ]; then
    TIME_PIN_NAME="PINNED_TIME_LIMIT_CHUNK_${ARM}"
  elif [ "$EXPECTED_STEP" -gt 0 ]; then
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
  NAME="FLAC_exp11_${ARM}"; EXPNAME="exp11_${ARM}"; SAVEDIR="${OUTPUT_ROOT}/exp11_${ARM}"
fi

case "$RUNG" in
  32x2|16x4|8x8) ;;
  *) die "rung '${RUNG}' must be 32x2, 16x4 or 8x8 - abort" ;;
esac
MB="${RUNG%x*}"; NGPU="${RUNG#*x}"
[ "$((MB * NGPU))" -eq 64 ] || die "rung ${RUNG}: MB*NGPU = $((MB*NGPU)) != 64 (micro x N pin, plan §10) - abort"
if [ "$SMOKE" != "1" ]; then
  [ "$MB" = "$PINNED_MB" ] && [ "$NGPU" = "$PINNED_NGPU" ] || die "pin inconsistency: rung ${RUNG} vs PINNED_MB=${PINNED_MB}/PINNED_NGPU=${PINNED_NGPU} - abort"
  [ "$MAXSTEPS" = "100000" ] || die "PINNED_MAXSTEPS is ${MAXSTEPS}, the registered budget is 100000 - abort"
fi

# --- B2. CHUNK_END: where THIS job stops (round 5) ----------------------------
# The budget gate above is untouched on purpose: CHUNK_END is the per-job stop,
# PINNED_MAXSTEPS stays the campaign budget. STOPSTEP is what train.py is given.
# Fail-closed on every edge, because a chunk that does not land ON a checkpoint
# leaves the chain with nothing to resume.
STOPSTEP="$MAXSTEPS"
if [ -n "$CHUNK_END" ]; then
  [ "$SMOKE" != "1" ] || die "CHUNK_END is a production chunk-chain input and has no meaning under SMOKE=1 (SMOKE_MAXSTEPS is the smoke's budget) - abort"
  case "$CHUNK_END" in ''|*[!0-9]*) die "CHUNK_END '${CHUNK_END}' must be a positive integer - abort";; esac
  [ "$EXPECTED_STEP" -gt 0 ] || die "CHUNK_END ${CHUNK_END} declares a chunk, but only a RESTART leg may be chunked (EXPECTED_STEP is 0) - abort"
  [ "$CHUNK_END" -gt "$EXPECTED_STEP" ] || die "CHUNK_END ${CHUNK_END} must exceed the resume step ${EXPECTED_STEP} — this leg would produce no new checkpoint - abort"
  [ "$CHUNK_END" -le "$MAXSTEPS" ] || die "CHUNK_END ${CHUNK_END} exceeds the budget ${MAXSTEPS} - abort"
  [ "$((CHUNK_END % PINNED_CHECKPOINT_EVERY))" -eq 0 ] || die "CHUNK_END ${CHUNK_END} is not a multiple of the ${PINNED_CHECKPOINT_EVERY}-step checkpoint cadence: the chunk must end ON a checkpoint or the next chunk has nothing to resume - abort"
  STOPSTEP="$CHUNK_END"
fi
RUNDIR="${SAVEDIR}/${NAME}/${EXPNAME}"
echo "=== exp_11 arm ${ARM} @ rung ${RUNG} (MB ${MB} x ${NGPU} GPU, grad-ckpt ON) — ${TS} — host $(hostname) ==="

# --- C. commit binding + tracked-surface drift --------------------------------
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || HEAD_SHA=""
EXPREL="${EXPDIR#"$REPO"/}"; EXP07REL="${EXP07#"$REPO"/}"
# The drift gate is scoped to CODE surfaces, not the whole exp folder: the four
# arms are running and Slurm appends to their tracked *.out logs continuously, so
# a folder-wide check would abort every screen on a live-log write. Configs,
# drivers and validators are still fully covered. The patterns are QUOTED so
# git, not the shell, expands them — a tracked file deleted from the worktree
# still matches (content-gate review B2) — data/AR (the split JSONs the
# dataloader opens) is covered, and a failing git status is fail-closed.
# arm_launch_registry.json is EXCLUDED (round-5 review B1): it is a lineage
# RECORD the reviewed recorder writes as each chunk finishes, not a measurement
# surface, and it is already outside the commit-binding closure below. Gating it
# deadlocks the chunk chain — the recorder dirties it at 42500 and every later
# leg, on every arm, would abort here until a human committed.
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
          "$EXPREL/*.json" "$EXPREL/*.py" "$EXPREL/*.sbatch" "$EXPREL/*.sh" \
          ":(exclude)$EXPREL/arm_launch_registry.json" \
          "$EXP07REL/FLAC_AR_BF.json" 2>&1)" \
  || die "git status for the drift gate failed: ${DRIFT} - abort"
# Commit binding is CONTENT-scoped: HEAD identity is sufficient but not
# necessary. Two sessions commit to this checkout, so a pending leg must
# survive commits that leave the training closure untouched — and abort on
# any commit that changes it. The closure is what the job actually loads:
# train.py, defaults.ini, src/, the data/AR split JSONs, the five arm
# configs (enumerated — a shell glob would silently drop a config deleted
# since EXPECT_SHA), this launcher, the four runtime helper scripts it
# invokes, and exp_07's FLAC_AR_BF.json (C4L parity baseline).
# Record/analysis files (registry, manifests, gen_*/validators, worklog)
# are deliberately OUTSIDE the closure. Fail-closed on every edge:
# EXPECT_SHA must be the full 40-hex commit OID (a symbolic ref like HEAD
# would defeat the binding), the diff runs against the CAPTURED HEAD OID,
# and HEAD is re-read afterwards to close the mid-gate-commit race.
# src/tests is EXCLUDED from both gates: pytest-only code, never imported by
# train.py at runtime, and the TDD sessions land test files continuously —
# gating them reproduced the exact kill this gate exists to prevent
# (legs 3680875-78 died on a src/tests-only commit, 2026-08-12).
surface_diff_vs_expect() {
  git diff --name-only "${EXPECT_SHA}" "${HEAD_SHA}" -- train.py defaults.ini src ":(exclude)src/tests" data/AR \
      "$EXPDIR"/FLAC_AR_BF_C4L.json "$EXPDIR"/FLAC_AR_BF_C8.json \
      "$EXPDIR"/FLAC_AR_BF_C16.json "$EXPDIR"/FLAC_AR_BF_C32.json \
      "$EXPDIR"/FLAC_AR_VANCKPT.json "$EXPDIR"/fa_orbit_train.sbatch \
      "$EXPDIR"/fa_orbit_ckpt_preflight.py "$EXPDIR"/assert_arm_configs_exp11.py \
      "$EXPDIR"/fa_orbit_wandb_readback.py "$EXPDIR"/fa_orbit_classify.py \
      "$EXP07/FLAC_AR_BF.json"
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
  [ -z "$DRIFT" ] || echo "DRY-RUN ADVISORY: tracked measurement surfaces are modified (a real launch aborts here)"
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "a real launch must run under sbatch (no SLURM_JOB_ID) - abort"
  [ -z "$GATE_FAIL" ] || die "${GATE_FAIL} - abort"
  echo "$GATE_OK_MSG"
  [ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
fi

# --- D. arm -> config (single source) + semantic gate -------------------------
arm_config_for() {
  case "$1" in
    C4L|C8|C16|C32) echo "$EXPDIR/FLAC_AR_BF_$1.json" ;;
    VANL)           echo "$EXPDIR/FLAC_AR_VANCKPT.json" ;;
    *) return 1 ;;
  esac
}
MODEL_CONFIG="$(arm_config_for "$ARM")" || die "no config mapped for arm '${ARM}' - abort"
MODEL_CONFIG_ABS="$(readlink -f "$MODEL_CONFIG" 2>/dev/null)"
[ -n "$MODEL_CONFIG_ABS" ] && [ -f "$MODEL_CONFIG_ABS" ] || die "arm config '${MODEL_CONFIG}' does not exist - abort"
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG_ABS" | awk '{print $1}')"
echo "config for ${ARM}: ${MODEL_CONFIG_ABS} sha256 ${CONFIG_SHA}"

python3 - "$MODEL_CONFIG_ABS" "$ARM" <<'PY' || die "arm/config semantic gate FAILED - abort"
import json, sys
cfg = json.load(open(sys.argv[1])); arm = sys.argv[2]
t = cfg.get("training", {}); bad = []
# VANL is the same recipe with the conditioning removed, so its gate is the
# MIRROR IMAGE of the orbit arms': the orbit keys must be ABSENT, not merely
# different. A vanilla config that carried a stray frame_avg_angles would be a
# silently fa-flavoured baseline, which would destroy the single-delta claim.
if arm == "VANL":
    cm = t.get("cond_method")
    if cm not in (None, "vanilla"):
        bad.append(f"cond_method={cm!r} (want absent or 'vanilla')")
    if "frame_avg_angles" in t:
        bad.append(f"frame_avg_angles is present ({t['frame_avg_angles']!r}) — a vanilla arm has no orbit")
    want = None
else:
    want = {"C4L": 4, "C8": 8, "C16": 16, "C32": 32}[arm]
    angles = t.get("frame_avg_angles")
    if t.get("cond_method") != "fa_invariant":
        bad.append(f"cond_method={t.get('cond_method')!r} (want fa_invariant)")
    if not isinstance(angles, list) or len(angles) != want:
        bad.append(f"frame_avg_angles has {angles and len(angles)} entries (want {want})")
    elif angles != [k * 360.0 / want for k in range(want)]:
        bad.append(f"frame_avg_angles are not the uniform C{want} orbit")
if t.get("use_ema") is not True:
    bad.append(f"use_ema={t.get('use_ema')!r} (want True)")
vits = [c for c in cfg["model"]["conditioning"]["configs"] if c["type"] == "ViTCoordinates"]
if sorted(c["id"] for c in vits) != ["context_poses_vit", "source_vit"]:
    bad.append(f"ViT conditioner ids {sorted(c['id'] for c in vits)} != the expected two")
# Post-P0: grad-ckpt ON for every arm; the KEY must exist and be literally True
for c in vits:
    if "gradient_checkpointing" not in c["config"]:
        bad.append(f"{c['id']}: gradient_checkpointing key absent (want literal true)")
    elif c["config"]["gradient_checkpointing"] is not True:
        bad.append(f"{c['id']}: gradient_checkpointing={c['config']['gradient_checkpointing']!r} (want True)")
if bad:
    sys.exit("ARM/CONFIG GATE: " + "; ".join(bad))
if arm == "VANL":
    print(f"gate OK: {arm} is vanilla (no cond_method, no orbit), grad-ckpt True, EMA on")
else:
    print(f"gate OK: {arm} carries the uniform C{want} orbit, grad-ckpt True, EMA on")
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
  [ "$MAXSTEPS" -gt "$EXPECTED_STEP" ] || die "MAXSTEPS ${MAXSTEPS} must exceed the resume step ${EXPECTED_STEP} - abort"
fi
echo "lineage: ${MODE} (expected_step ${EXPECTED_STEP}, max_steps ${MAXSTEPS}, chunk_end ${CHUNK_END:-<none>}, stops at ${STOPSTEP}, ckpt every ${CHECKPOINT_EVERY}, time pin ${TIME_PIN_NAME}=${TIME_LIMIT})"

# --- F. the exact train.py argv ----------------------------------------------
ARGV=(
  --model-config "$MODEL_CONFIG_ABS"
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps "$STOPSTEP" --batch-size "$MB" --accum-batches 1 --num-workers 6 --seed 42
  --num-gpus "$NGPU" --num-nodes 1
  --strategy ddp_find_unused_parameters_true --sync-batchnorm true --precision bf16-mixed
  --val-every -1 --val-dataset-config ''
  --gradient-clip-val 0.0
  --logger wandb --checkpoint-every "$CHECKPOINT_EVERY"
  --name "$NAME" --experiment-name "$EXPNAME" --save-dir "$SAVEDIR"
)
[ "$MODE" = "RESTART" ] && ARGV+=(--ckpt-path "$RESUME_CKPT")

# --- G. argv-parity dry run (plan N13; round-3 N9 tightened) ------------------
ARGV_FILE="$(mktemp)" || die "mktemp failed - abort" 3
printf '%s\n' "${ARGV[@]}" > "$ARGV_FILE" || die "could not write the argv file - abort" 3
python3 - "$ARGV_FILE" "$MODE" <<'PY'
import sys
# The exp_07 B-F reference argv (bf_scratch_launch.sh) — the lineage this sweep continues.
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
                f"{flag}: exp_07 {ref[flag]!r} -> exp_11 {new[flag]!r}")
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
  echo "  (Slurm/GPU/VRAM/env/wandb/ViT/lock gates and training are skipped in DRYRUN)"
  exit 0
fi

# --- H. Slurm allocation must match the pins (round-3 B1) ---------------------
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
[ "${SLURM_NTASKS:-1}" = "1" ] || die "expected 1 task, got ${SLURM_NTASKS} - abort"
WANT_CPUS="$((8 + 7 * NGPU))"; WANT_MEM_MB="$(((12 * NGPU + 12) * 1024))"
GOT_CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-0}}"
GOT_MEM_MB="${SLURM_MEM_PER_NODE:-0}"
[ "$GOT_CPUS" = "$WANT_CPUS" ] || die "allocated ${GOT_CPUS} CPUs, the pinned rung needs ${WANT_CPUS} — submit via fa_orbit_submit.sh - abort"
[ "$GOT_MEM_MB" = "$WANT_MEM_MB" ] || die "allocated ${GOT_MEM_MB} MB RAM, the pinned rung needs ${WANT_MEM_MB} — submit via fa_orbit_submit.sh - abort"
GOT_TIME="$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ')"
norm_minutes() { awk -v t="$1" 'BEGIN{d=0; if (t ~ /-/) {split(t,p,"-"); d=p[1]; t=p[2]} n=split(t,c,":");
  if (n==3) m=c[1]*60+c[2]+c[3]/60; else if (n==2) m=c[1]+c[2]/60; else m=t; printf "%d", d*1440+m}'; }
# The pin this ${MODE} leg is entitled to — an INITIAL allocation handed to a
# RESTART leg (or the reverse) is refused here, in the job, not merely intended
# by the submitter.
[ "$(norm_minutes "$GOT_TIME")" = "$(norm_minutes "$TIME_LIMIT")" ] || die "allocated time ${GOT_TIME} != the ${TIME_PIN_NAME} pin ${TIME_LIMIT} this ${MODE} leg requires — submit via fa_orbit_submit.sh - abort"
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
LOCKFILE="${OUTPUT_ROOT}/exp11_${ARM}.lock"
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
  # Q10 / re-pin fix 1: a real arm's restart is the 40k -> 100k EXTENSION, whose
  # contract binds the ORIGINAL launch identity (audited manifest bytes, job,
  # uuid, launch commit, config, save-dir, seed, and the 40k anchor itself)
  # without demanding that the INITIAL budget/commit equal this leg's. SMOKE
  # restarts have no registered launch and keep the ordinary contract.
  #
  # Round 5: only the FIRST leg off the anchor resumes the audited 40k checkpoint.
  # A later chunk resumes the endpoint of the previous chunk, which is evidence
  # only once fa_orbit_record_restart.py has written it as a chain link — so those
  # legs bind with --chain, and an unrecorded predecessor refuses the launch.
  if [ "$SMOKE" != "1" ]; then
    if [ "$EXPECTED_STEP" -eq 40000 ]; then
      PRE_ARGS+=(--extension)
    else
      PRE_ARGS+=(--chain)
    fi
    PRE_ARGS+=(--launch-registry "$EXPDIR/arm_launch_registry.json" --repo-root "$REPO")
    [ -n "$CHUNK_END" ] && PRE_ARGS+=(--chunk-end "$CHUNK_END")
  fi
  PRE_OUT="$(python3 "$EXPDIR/fa_orbit_ckpt_preflight.py" "${PRE_ARGS[@]}" 2>&1)"
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
  ORIG_WANDB_RUN_ID="$(awk '/^wandb_run_id /{print $2}' "$LAUNCH_MANIFEST_LINK" 2>/dev/null)"
  [ -n "$ORIG_WANDB_RUN_ID" ] || die "RESTART needs the original wandb run id from ${LAUNCH_MANIFEST_LINK} - abort"
  # A RESUMED wandb run refuses config changes: prefigure's push_wandb_config
  # calls config.update() without allow_val_change, and a restart legitimately
  # changes max_steps (40000->100000) and ckpt_path — legs 3684149/3684150
  # died at exactly train.py:193 on this. So a restart runs as a FRESH run
  # carrying the lineage in its name; curves split across runs but panels
  # align on the step axis. train.py stays byte-unchanged (exp_15 admission
  # pins it).
  WANDB_RUN_ID="exp11-${ARM}-r${EXPECTED_STEP}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
  export WANDB_RUN_ID
  echo "wandb: new RESTART run id ${WANDB_RUN_ID} (continues ${ORIG_WANDB_RUN_ID}; not resumed — a resumed run refuses the changed max_steps/ckpt_path config)"
else
  WANDB_RUN_ID="exp11-${ARM}-$(date +%s%N)-$(cut -c1-8 /proc/sys/kernel/random/uuid)"
  export WANDB_RUN_ID
  echo "wandb: new run id ${WANDB_RUN_ID}"
fi

# --- N. DINOv3 pin + init-identity gate (inside the allocation) ---------------
HF_HUB_OFFLINE=1 python3 "$EXPDIR/assert_arm_configs_exp11.py" "$ARM" || die "ViT-pin / init-identity GATE FAILED - abort" 1

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

TRAINLOG="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
SAVEDIR_LOG="${SAVEDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_train.log"
MANIFEST="${EXPDIR}/fa_orbit_${TS}_${ARM}_${RUNG}_jid${SLURM_JOB_ID}_manifest.txt"
# B5 residual: a failed environment dump must not be silently hashed into the record
PIPFREEZE_FILE="$(mktemp)" || die "mktemp failed - abort" 3
pip freeze > "$PIPFREEZE_FILE" 2>/dev/null || die "pip freeze FAILED - the environment digest would be a lie - abort" 7
ENV_SHA="$(sha256sum "$PIPFREEZE_FILE" | awk '{print $1}')"
rm -f "$PIPFREEZE_FILE"
{
  echo "# exp_11 arm launch manifest"
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
  echo "chunk_end ${CHUNK_END:-<none>}"
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

wait "$TR_PID"; rc=$?
kill "$WATCHER" 2>/dev/null; wait "$WATCHER" 2>/dev/null
wait "$TEE_PID"; tee_rc=$?
rm -f "$FIFO"

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
python3 "$EXPDIR/fa_orbit_wandb_readback.py" --run-id "$WANDB_RUN_ID" \
  --root "$REPO" --root "$WANDB_DIR" \
  --entity "$WANDB_ENTITY" --project "$NAME" --name "$EXPNAME" || WANDB_CHECK_RC=$?
if [ "$WANDB_CHECK_RC" -ne 0 ]; then
  echo "W&B run identity could not be verified against the manifest - provenance failure"
fi

# --- R. classification (tested taxonomy; see fa_orbit_classify.py) ------------
# The completion literal Lightning prints is the stop step it was GIVEN, so the
# classifier is told the step this leg was asked to reach. Without a chunk that
# is MAXSTEPS, byte-identical to every pre-round-5 leg; with one it is the chunk
# boundary, and a successful chunk classifies COMPLETE instead of NO-MARKER.
CLASSIFY_OUT="$(python3 "$EXPDIR/fa_orbit_classify.py" --rc "$rc" --tee-rc "$tee_rc" --ngpu "$NGPU" \
  --maxsteps "$STOPSTEP" --log "$TRAINLOG" --log-copy "$SAVEDIR_LOG" 2>&1)"
final_rc=$?
echo "$CLASSIFY_OUT"
LAST_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*.ckpt 2>/dev/null | head -1)"

# --- S. durable final record (round-3 B5 residual) ---------------------------
# The classifier verdict and this summary must survive in BOTH copies; the tee
# status is captured (it used to be discarded) and a failure is class 7. The
# whole preflight transcript (this job's Slurm output so far) is copied into the
# save-dir too, so the durable record is not just torchrun's stdout.
FINAL_RECORD="$(
  echo "=== exp_11 ${ARM} ${RUNG} finished at $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "torchrun rc=${rc} tee rc=${tee_rc} wandb_identity_rc=${WANDB_CHECK_RC} classified rc=${final_rc}"
  echo "$CLASSIFY_OUT"
  echo "last checkpoint: ${LAST_CKPT:-<none>}"
  [ -n "$CHUNK_END" ] && echo "endpoint attestation: appended BELOW this record, once the exit class is settled (round-5 r2 blocking 2)"
  echo "restart: fa_orbit_submit.sh ${ARM} --resume ${LAST_CKPT:-<path>} --expected-step <step>"
  [ -n "$CHUNK_END" ] && echo "next chunk: record this leg (fa_orbit_record_restart.py ${ARM} ${MANIFEST}), then fa_orbit_submit.sh ${ARM} --resume <ckpt at ${CHUNK_END}> --expected-step ${CHUNK_END} --chunk-end <${CHUNK_END}+${PINNED_CHECKPOINT_EVERY}>"
)"
printf '%s\n' "$FINAL_RECORD" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null
final_tee_rc="${PIPESTATUS[1]}"
# The transcript path is NOT guessed from the #SBATCH pattern (job 3646734 logged
# "slurm_train_job_3646734.out not found" because SLURM_JOB_NAME is unset inside
# the job): ask Slurm itself.
# --- BEGIN slurm-stdout-helper (guard-tested by fa_orbit_equivprobe_wrapper_test.sh) --
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
# ^^^ THE EXIT CLASS IS SETTLED HERE. Nothing below may change final_rc. ^^^

# --- R2. ENDPOINT ATTESTATION: the job attests what it produced --------------
# (round-5 B6, moved here by the round-5 r2 review, blockings 2 and 3.)
#
# The manifest is published BEFORE training, so on its own it proves only what
# this job intended. The recorder would otherwise have to associate it with
# whichever exactly-named checkpoint happened to sit in the shared directory — a
# failed job's manifest plus a pre-existing same-arm checkpoint could become a
# chain link, and re-hashing proves current bytes, not authorship. So the job
# itself appends what it produced: the endpoint path, its step, and its sha256 as
# this job sees it on disk. fa_orbit_record_restart.py REFUSES any chunk manifest
# without this attestation, one attesting `<none>`, and one whose attested sha
# does not match its own re-hash.
#
# BLOCKING 2 — WHERE this runs is load-bearing. It used to run right after the
# classifier, where final_rc=0 was only PROVISIONAL: the final-record tee, the
# preflight-copy and the W&B identity check can each still promote the class to 7
# afterwards, so a job that ended up FAILED could carry a positive attestation.
# It now runs after the settle point above, so a positive attestation is emitted
# ONLY for the success class (final_rc == 0); every other class attests <none>.
#
# BLOCKING 3 — WHICH FILE is load-bearing too. train.py builds ModelCheckpoint
# with Lightning's version counter at its default (enable_version_counter=True),
# so a retry at the same boundary writes `epoch=E-step=N-v1.ckpt` beside the
# unversioned file a previous failed attempt left behind. Globbing only the
# unversioned name would attest the STALE bytes — another job's checkpoint. Both
# name shapes are therefore searched and the NEWEST-mtime match wins: that is the
# file THIS job just wrote. Its actual path is what is attested and recorded.
ATTEST_LINE=""
if [ -n "$CHUNK_END" ]; then
  ENDPOINT_CKPT=""
  if [ "$final_rc" -eq 0 ]; then
    ENDPOINT_CKPT="$(ls -1t "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}".ckpt \
                              "${RUNDIR}/checkpoints"/*-step="${CHUNK_END}"-v*.ckpt 2>/dev/null | head -1)"
  fi
  if [ "$final_rc" -ne 0 ]; then
    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
    echo "endpoint attestation: this leg's SETTLED class is ${final_rc}, not the success class — attesting <none>"
  elif [ -z "$ENDPOINT_CKPT" ] || [ ! -f "$ENDPOINT_CKPT" ]; then
    ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
    echo "endpoint attestation: no *-step=${CHUNK_END}.ckpt or *-step=${CHUNK_END}-v*.ckpt under ${RUNDIR}/checkpoints — attesting <none>"
  else
    ENDPOINT_SHA="$(sha256sum "$ENDPOINT_CKPT" | awk '{print $1}')"
    if [ -n "$ENDPOINT_SHA" ]; then
      ATTEST_LINE="endpoint_ckpt ${ENDPOINT_CKPT} endpoint_step ${CHUNK_END} endpoint_sha256 ${ENDPOINT_SHA}"
      echo "endpoint attestation: ${ENDPOINT_CKPT} step ${CHUNK_END} sha256 ${ENDPOINT_SHA} (newest match; settled class ${final_rc})"
    else
      ATTEST_LINE="endpoint_ckpt <none> endpoint_step ${CHUNK_END} endpoint_class ${final_rc}"
      echo "endpoint attestation: sha256sum of ${ENDPOINT_CKPT} FAILED — attesting <none>"
    fi
  fi
  # The manifest was written pre-run by THIS job; this is the same job appending
  # to its own record. Both copies are kept identical.
  if ! printf '%s\n' "$ATTEST_LINE" >> "$MANIFEST"; then
    echo "ENDPOINT ATTESTATION COULD NOT BE APPENDED to ${MANIFEST} — the recorder will refuse this chunk (fail-closed); record it by hand only after auditing the checkpoint"
  fi
  printf '%s\n' "$ATTEST_LINE" >> "${SAVEDIR}/$(basename "$MANIFEST")" \
    || echo "WARNING: could not append the endpoint attestation to the save-dir manifest copy"
  # ...and into BOTH durable logs, so the transcript carries what the manifest
  # carries. The classifier's log-equality check already ran; appending the same
  # bytes to both copies keeps them identical.
  printf 'endpoint attestation: %s\n' "$ATTEST_LINE" | tee -a "$TRAINLOG" "$SAVEDIR_LOG" > /dev/null \
    || echo "WARNING: could not append the endpoint attestation to the durable logs"
fi
printf '%s\n' "$FINAL_RECORD"
exit "$final_rc"
