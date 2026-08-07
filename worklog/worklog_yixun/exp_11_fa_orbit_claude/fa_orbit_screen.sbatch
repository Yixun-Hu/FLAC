#!/bin/bash
# ============================================================================
# exp_11 SCREEN DRIVER — one checkpoint, one arm, own protocol (plan §3/§4/§5).
#
# Futility-gate screens (plan §4) and the C4 backfill (plan §3) run through this
# single driver so they cannot drift apart:
#
#   ARM=C4L|C8|C16|C32   the arm's OWN checkpoint at STEP, evaluated under the
#                        arm's OWN orbit (announcement 04: each model under its
#                        own protocol), eval-name exp11_<ARM>_screen_S<STEP>_s<SEED>_K<K>
#   ARM=C4BACKFILL       the exp_07 B-F lineage at STEP in {10000,20000,30000},
#                        C4 orbit, eval-name exp11_C4backfill_S<STEP>_s<SEED>_K<K>
#                        — the fa-protocol backfill the futility gates compare
#                        against (the exp_07 S10000-S30000 screens were the
#                        retracted mismatched-protocol evals).
#
# Protocol, matching exp_07/exp_10: full unseen split, EMA weights (the arm
# config), cfg 1.0, 1 diffusion step, bf16 conditioning autocast, seed 42 by
# default, K=8 by default (K=1 uses the _1 split).
#
# Fail-closed before spending the GPU: commit binding + tracked drift, exactly
# one visible L40, EXACTLY ONE checkpoint matching the step, and the
# checkpoint's OWN embedded model_config must carry this arm's orbit (a screen
# that evaluated the wrong arm's checkpoint would poison a futility gate).
# Afterwards it writes the sidecar that exp11_validate_rows.py requires and
# emits one SCREENRESULT line.
#
# Submit (resources are in the directives; CLI flags still override):
#   sbatch --job-name=exp11-screen-C4L-10000 \
#          --export=ALL,EXPECT_SHA=$(git rev-parse HEAD),ARM=C4L,STEP=10000 \
#          worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch
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
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_screen_%x_%j.out

set -uo pipefail
REPO=/n/fs/gatrdp/codespace/FLAC
# TEST HOOK (guard tests only; ignored inside a Slurm job, like the launcher's):
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
SEED="${SEED:-42}"
K="${K:-8}"
# CELL selects the purpose-specific contract (round-4 review B4). 'screen' is the
# futility cadence: seed 42, K=8, single seed by design. 'conf' is a confirmatory
# table cell: seeds 42-46, K in {1,8}. The driver refuses combinations the
# validator's contracts would reject, instead of labelling everything 'screen'.
CELL="${CELL:-screen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_FLAC}"
PRODUCTION_OUTPUT_ROOT="outputs_FLAC"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "${OUTPUT_ROOT:-}" ] && [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ]; then
    echo "ambient OUTPUT_ROOT='${OUTPUT_ROOT}' != '${PRODUCTION_OUTPUT_ROOT}' - abort"; exit 2
  fi
  OUTPUT_ROOT="$PRODUCTION_OUTPUT_ROOT"
fi
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

die() { echo "$1"; exit "${2:-2}"; }

# --- A. parameters ------------------------------------------------------------
[ -n "${ARM:-}" ] || die "ARM must be exported (C4L|C8|C16|C32|C4BACKFILL) - abort"
[ -n "${STEP:-}" ] || die "STEP must be exported - abort"
[ -n "${EXPECT_SHA:-}" ] || die "EXPECT_SHA (full reviewed commit sha) must be exported - abort"
case "$ARM" in
  C4L|C8|C16|C32|C4BACKFILL) ;;
  *) die "ARM '${ARM}' is not screenable (C4L|C8|C16|C32|C4BACKFILL) - abort" ;;
esac
case "$STEP" in ''|*[!0-9]*) die "STEP '${STEP}' must be a positive integer - abort";; esac
[ "$STEP" -gt 0 ] || die "STEP must be > 0 - abort"
case "$SEED" in ''|*[!0-9]*) die "SEED '${SEED}' must be an integer - abort";; esac
case "$K" in 1|8) ;; *) die "K '${K}' must be 1 or 8 (the registered eval schedules) - abort";; esac
case "$CELL" in
  screen)
    [ "$SEED" = "42" ] || die "a futility screen is seed 42 by contract (got ${SEED}); use CELL=conf for the 5-seed table cell - abort"
    [ "$K" = "8" ] || die "a futility screen is K=8 by contract (got K=${K}) - abort" ;;
  conf)
    case "$SEED" in 42|43|44|45|46) ;; *) die "a confirmatory cell uses seeds 42-46 (got ${SEED}) - abort";; esac ;;
  *) die "CELL '${CELL}' must be 'screen' (futility) or 'conf' (confirmatory table cell) - abort" ;;
esac
if [ "$ARM" = "C4BACKFILL" ] && [ "$CELL" != "screen" ]; then
  die "the C4 backfill is a futility-gate comparator only (CELL=screen) - abort"
fi
if [ "$ARM" = "C4BACKFILL" ]; then
  # The plan pre-registers backfill gates at 20k and 30k ONLY (§3/§4). A 10k
  # backfill cell would look like registered evidence and is not (round-4 NIT 2);
  # the arms' own 10k screens remain fine — those are registered screen cadence.
  case "$STEP" in
    20000|30000) ;;
    *) die "the C4 backfill is registered at steps 20000/30000 only (plan §3), got ${STEP} - abort" ;;
  esac
fi

# --- B. arm -> config, orbit, checkpoint tree, identity -----------------------
case "$ARM" in
  C4BACKFILL)
    MODEL_CONFIG="$EXP07/FLAC_AR_BF.json"
    CKPT_DIR="${OUTPUT_ROOT}/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints"
    N_ANGLES=4
    EVAL_NAME="exp11_C4backfill_S${STEP}_s${SEED}_K${K}"
    ;;
  *)
    MODEL_CONFIG="$EXPDIR/FLAC_AR_BF_${ARM}.json"
    CKPT_DIR="${OUTPUT_ROOT}/exp11_${ARM}/FLAC_exp11_${ARM}/exp11_${ARM}/checkpoints"
    case "$ARM" in C4L) N_ANGLES=4;; C8) N_ANGLES=8;; C16) N_ANGLES=16;; C32) N_ANGLES=32;; esac
    EVAL_NAME="exp11_${ARM}_${CELL}_S${STEP}_s${SEED}_K${K}"
    ;;
esac
[ -f "$MODEL_CONFIG" ] || die "model config not found: ${MODEL_CONFIG} - abort"
CONFIG_SHA="$(sha256sum "$MODEL_CONFIG" | awk '{print $1}')" || die "sha256sum failed - abort"
ANGLES="$(python3 -c "print(','.join(repr(k*360.0/$N_ANGLES) for k in range($N_ANGLES)))")" \
  || die "could not build the orbit for ${ARM} - abort"
case "$K" in
  8) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json" ;;
  1) DATASET_CONFIG="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json" ;;
esac
[ -f "$DATASET_CONFIG" ] || die "eval dataset config not found: ${DATASET_CONFIG} - abort"
echo "=== exp_11 screen ${ARM} S${STEP} s${SEED} K${K} — ${TS} — host $(hostname) ==="
echo "config ${MODEL_CONFIG} sha256 ${CONFIG_SHA} | orbit C${N_ANGLES} | eval-name ${EVAL_NAME}"

# --- C. commit binding + tracked drift ---------------------------------------
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED - abort"
# The drift gate stays RECURSIVE over the experiment directory (so future
# subdirectory helpers, manifests and decision-relevant markdown remain covered)
# and excludes only the append-only live artifacts the running arms write:
# their Slurm transcripts and this driver's own screen logs (round-4 NIT 1).
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src eval_FLAC.py \
          "$EXPDIR" ":(exclude)$EXPDIR/slurm_train_*.out" ":(exclude)$EXPDIR/slurm_screen_*.out" \
          ":(exclude)$EXPDIR/*_screen.log")" \
  || die "git status FAILED - refusing to treat an error as a clean tree - abort"
if [ "$DRYRUN" = "1" ]; then
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] && echo "commit binding OK (dry run): ${HEAD_SHA}" \
    || echo "DRY-RUN ADVISORY: HEAD != EXPECT_SHA (a real screen aborts here)"
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "a real screen must run under sbatch - abort"
  [ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"
  [ -z "$DRIFT" ] || { echo "tracked surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
  echo "commit binding OK: ${HEAD_SHA}"
fi

# --- D. exactly ONE checkpoint at this step ----------------------------------
mapfile -t CKPTS < <(ls -1 "${CKPT_DIR}"/epoch=*-step=${STEP}.ckpt 2>/dev/null)
[ "${#CKPTS[@]}" -eq 1 ] || die "expected exactly 1 checkpoint at step ${STEP} under ${CKPT_DIR}, found ${#CKPTS[@]} - abort"
CKPT="${CKPTS[0]}"
echo "checkpoint: ${CKPT}"

# --- D2. backfill only: the checkpoint must be the AUDITED one (review B6) ----
# Parsed-config equality proves the recipe, not the file. For the historical
# lineage the plan requires an audited seed-to-file manifest, so the backfill
# checkpoint is bound to its recorded path AND sha256.
if [ "$ARM" = "C4BACKFILL" ]; then
  BACKFILL_MANIFEST="$EXPDIR/c4_backfill_manifest.json"
  [ -f "$BACKFILL_MANIFEST" ] || die "backfill manifest missing: ${BACKFILL_MANIFEST} - abort"
  python3 - "$BACKFILL_MANIFEST" "$STEP" "$CKPT" <<'PY' || die "backfill lineage gate FAILED - abort"
import hashlib, json, os, sys
man, step, ckpt = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
entry = man["checkpoints"].get(step)
if entry is None:
    sys.exit(f"step {step} is not in the audited backfill manifest "
             f"(registered: {sorted(man['checkpoints'])})")
if os.path.realpath(entry["path"]) != os.path.realpath(ckpt):
    sys.exit(f"resolved checkpoint {ckpt} != the audited {entry['path']}")
h = hashlib.sha256()
with open(ckpt, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != entry["sha256"]:
    sys.exit(f"checkpoint sha256 {h.hexdigest()} != the audited {entry['sha256']}")
print(f"backfill lineage OK: step {step} matches the audited manifest "
      f"({entry['sha256'][:12]}..., seed {man['training_seed']})")
PY
fi

# --- D3. arms only: bind the checkpoint to its arm's LAUNCH MANIFEST ---------
# Discovery + embedded-config equality proves the recipe, not the run: any
# checkpoint with a matching config dropped into the tree would pass. The arm's
# own launch manifest (written by fa_orbit_train.sbatch at INITIAL launch)
# records arm, rung, seed, config hash and canonical save-dir; the checkpoint
# must sit in exactly that run directory and the config must be that manifest's
# (re-review item 2).
if [ "$ARM" != "C4BACKFILL" ]; then
  ARM_LAUNCH_MANIFEST="${OUTPUT_ROOT}/exp11_${ARM}/launch_manifest.txt"
  [ -f "$ARM_LAUNCH_MANIFEST" ] || die "arm launch manifest missing: ${ARM_LAUNCH_MANIFEST} — a screen may only evaluate a checkpoint from a recorded launch - abort"
  python3 - "$ARM_LAUNCH_MANIFEST" "$ARM" "$CKPT" "$CONFIG_SHA" "$CKPT_DIR" <<'PY' || die "arm launch-manifest lineage gate FAILED - abort"
import os, sys
man_path, arm, ckpt, cfg_sha, ckpt_dir = sys.argv[1:6]
man = {}
for line in open(man_path):
    line = line.strip()
    if line and not line.startswith("#"):
        k, _, rest = line.partition(" ")
        man[k] = rest.strip()
fields = ("arm " + man.get("arm", "")).split()
kv = {fields[i]: fields[i + 1] for i in range(0, len(fields) - 1, 2)}
bad = []
if kv.get("arm") != arm:
    bad.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
if man.get("config_sha256") != cfg_sha:
    bad.append(f"manifest config_sha256 {man.get('config_sha256', '')[:12]} != the config we would "
               f"evaluate {cfg_sha[:12]}")
save_dir = man.get("save_dir", "")
if not save_dir:
    bad.append("manifest records no save_dir")
else:
    want_dir = os.path.realpath(ckpt_dir)
    canon = os.path.realpath(os.path.join(save_dir, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
    if want_dir != canon:
        bad.append(f"checkpoint dir {want_dir} is not the manifest's canonical {canon}")
    if os.path.realpath(os.path.dirname(ckpt)) != canon:
        bad.append(f"checkpoint {ckpt} does not live in the canonical run directory {canon}")
if bad:
    sys.exit("ARM LINEAGE GATE: " + "; ".join(bad))
print(f"arm lineage OK: {arm} bound to launch {man.get('job', '?')} "
      f"(commit {man.get('commit', '?')[:12]}, rung {kv.get('rung')}, seed 42 recipe, "
      f"config {cfg_sha[:12]})")
PY
fi

# --- E. the checkpoint's OWN embedded config must be this arm's ---------------
python3 - "$CKPT" "$MODEL_CONFIG" "$N_ANGLES" "$STEP" "$EXPDIR" <<'PY' || die "checkpoint/arm identity gate FAILED - abort"
import json, sys
ckpt, cfg_path, n_angles, step, expdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, expdir)
from fa_orbit_ckpt_preflight import load_ckpt_config, load_ckpt_state_keys
cfg, gstep = load_ckpt_config(ckpt)
ck_state_keys = load_ckpt_state_keys(ckpt)
tr = cfg.get("training", {})
want = [k * 360.0 / n_angles for k in range(n_angles)]
bad = []
if gstep != step:
    bad.append(f"embedded global_step {gstep} != {step}")
if tr.get("cond_method") != "fa_invariant":
    bad.append(f"embedded cond_method={tr.get('cond_method')!r} != 'fa_invariant'")
sd = ck_state_keys
if not any(str(k).startswith("diffusion_ema.ema_model.") for k in sd):
    bad.append("checkpoint carries no diffusion_ema.ema_model.* weights — eval_FLAC would "
               "silently fall back to ONLINE weights and the row would not be an EMA row")
if tr.get("frame_avg_angles") != want:
    bad.append(f"embedded frame_avg_angles={tr.get('frame_avg_angles')!r} != the C{n_angles} orbit")
if cfg != json.load(open(cfg_path)):
    bad.append(f"embedded model_config != {cfg_path} (parsed-object mismatch)")
if bad:
    sys.exit("CKPT/ARM GATE: " + "; ".join(bad))
print(f"ckpt gate OK: step {gstep}, fa_invariant C{n_angles}, config matches {cfg_path}")
PY

# --- F. the eval argv ---------------------------------------------------------
EVAL_ARGV=(
  --model-config "$MODEL_CONFIG"
  --dataset-config "$DATASET_CONFIG"
  --ckpt-path "$CKPT"
  --cond-method fa_invariant --frame-avg-angles "$ANGLES"
  --cond-autocast bf16 --cfg-scale 1.0 --steps 1
  --seed "$SEED" --eval-name "$EVAL_NAME"
)
echo "--- eval argv ---"
echo "python eval_FLAC.py ${EVAL_ARGV[*]}"
if [ "$DRYRUN" = "1" ]; then
  echo "DRY RUN complete: gates A-E passed for ARM=${ARM} STEP=${STEP} SEED=${SEED} K=${K}"
  exit 0
fi

# --- G. GPU + run -------------------------------------------------------------
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits) \
  || die "nvidia-smi FAILED - abort"
[ "${#GPU_ROWS[@]}" -eq 1 ] || die "expected exactly 1 visible GPU, got ${#GPU_ROWS[@]} - abort"
case "$(echo "${GPU_ROWS[0]}" | cut -d, -f2)" in *L40*) ;; *) die "not an L40 - abort";; esac
echo "GPU: ${GPU_ROWS[0]}"

# --- BEGIN screen-run-and-record (guard-tested by fa_orbit_screen_guardtests.sh) --
LOG="${LOG:-${EXPDIR}/fa_orbit_${TS}_${ARM}_S${STEP}_s${SEED}_K${K}_screen.log}"
python eval_FLAC.py "${EVAL_ARGV[@]}" 2>&1 | tee "$LOG"
PIPE_RC=("${PIPESTATUS[@]}")
EVAL_RC="${PIPE_RC[0]}"; TEE_RC="${PIPE_RC[1]}"
[ "$TEE_RC" -eq 0 ] || die "tee exited ${TEE_RC} - the screen log is not durable - abort" 7
[ "$EVAL_RC" -eq 0 ] || die "eval_FLAC.py exited ${EVAL_RC} (see ${LOG})" "$EVAL_RC"

# --- H. sidecar + result line -------------------------------------------------
CKPT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')" || die "ckpt sha256 FAILED - abort"
METRICS="$(python3 - "$CKPT" "$EVAL_NAME" "$N_ANGLES" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from eval_FLAC import build_output_paths
print(build_output_paths(sys.argv[1], 1, 1.0, sys.argv[2], cond_method="fa_invariant",
                         n_angles=int(sys.argv[3]))["metrics"])
PY
)" || die "could not derive the metrics path - abort"
[ -f "$METRICS" ] || die "metrics JSON not written: ${METRICS} - abort" 5
python3 - "$METRICS" "$ARM" "$STEP" "$SEED" "$K" "$EVAL_NAME" "$MODEL_CONFIG" "$CONFIG_SHA" \
         "$DATASET_CONFIG" "$CKPT" "$CKPT_SHA" "$ANGLES" "$HEAD_SHA" <<'PY' || die "sidecar write FAILED - abort" 7
import json, sys
(metrics, arm, step, seed, k, eval_name, cfg, cfg_sha, ds, ckpt, ckpt_sha, angles, commit) = sys.argv[1:14]
side = {
    "arm": arm, "step": int(step), "seed": int(seed), "K": int(k), "eval_name": eval_name,
    "cfg_scale": 1.0, "steps": 1, "model_config": cfg, "model_config_sha256": cfg_sha,
    "dataset_config": ds, "ckpt_path": ckpt, "ckpt_sha256": ckpt_sha, "use_ema": True,
    "frame_avg_angles": [float(a) for a in angles.split(",")],
    "cond_method": "fa_invariant", "cond_autocast": "bf16", "commit": commit,
}
with open(metrics + ".screenmeta.json", "w") as fh:
    json.dump(side, fh, indent=2)
print("sidecar:", metrics + ".screenmeta.json")
PY

# --- I. the validator is a GATE, not advice (round-4 review B5) ---------------
# A screen that cannot be validated must not announce a result: SCREENRESULT is
# what the futility gate and the table generator consume.
# A confirmatory row cannot be validated alone -- the table contract is a
# five-seed transaction -- so it must NOT claim it was (re-review item 5). Only a
# futility screen, which IS a complete single-seed cell, reports validated=.
if [ "$CELL" = "screen" ]; then
  python3 "$EXPDIR/exp11_validate_rows.py" --arm "$ARM" --step "$STEP" --k "$K" \
    --contract futility --verify-hashes "$METRICS" \
    || die "row validation FAILED - refusing to emit SCREENRESULT for an unprovable row" 6
  VALIDATED="futility"
else
  VALIDATED="pending-cell"
  echo "NOTE: CELL=${CELL} is one row of a five-seed table cell; it is validated as a WHOLE cell"
  echo "  once every seed has landed, and until then this row is NOT validated:"
  echo "  python3 ${EXPDIR}/exp11_validate_rows.py --arm ${ARM} --step ${STEP} --k ${K} --contract table --verify-hashes <all seed JSONs>"
fi

echo "SCREENRESULT arm=${ARM} cell=${CELL} step=${STEP} seed=${SEED} k=${K} rc=${EVAL_RC} validated=${VALIDATED} eval_name=${EVAL_NAME} metrics=${METRICS} sidecar=${METRICS}.screenmeta.json config_sha256=${CONFIG_SHA} ckpt_sha256=${CKPT_SHA} commit=${HEAD_SHA} log=${LOG}"
# --- END screen-run-and-record -------------------------------------------------
exit 0
