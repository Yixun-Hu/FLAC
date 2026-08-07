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
# WORKTREE-PINNED EXECUTION (item 8): code comes from a worktree pinned to the
# submission SHA, outputs go to the main tree. Development and measurement are
# therefore DECOUPLED — the main checkout may keep committing while jobs run, and
# source_sha still equals the pinned SHA. MEASURE_ROOT is REQUIRED for a real run.
#
# Submit (resources are in the directives; CLI flags still override):
#   MEASURE_ROOT=$(bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh)
#   sbatch --job-name=exp11-screen-C4L-10000 \
#          --export=ALL,EXPECT_SHA=$(git -C $MEASURE_ROOT rev-parse HEAD),MEASURE_ROOT=$MEASURE_ROOT,ARM=C4L,STEP=10000 \
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
MAIN_REPO=/n/fs/gatrdp/codespace/FLAC        # where OUTPUTS and logs always live
REPO="$MAIN_REPO"
# TEST HOOK (guard tests only; ignored inside a Slurm job, like the launcher's):
if [ -n "${FA_ORBIT_REPO_OVERRIDE:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  REPO="$FA_ORBIT_REPO_OVERRIDE"; MAIN_REPO="$FA_ORBIT_REPO_OVERRIDE"
fi
unset FA_ORBIT_REPO_OVERRIDE

# --- WORKTREE-PINNED MEASUREMENT (item 8) ------------------------------------
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
EXPDIR="$CODE_ROOT/worklog/worklog_yixun/exp_11_fa_orbit_claude"     # CODE (pinned)
EXP07="$CODE_ROOT/worklog/worklog_yixun/exp_07_fa_scratch_claude"    # CODE (pinned)
LOGDIR="$MAIN_REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"     # LOGS (main tree)
cd "$CODE_ROOT" || exit 3
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_REPO/outputs_FLAC}"
PRODUCTION_OUTPUT_ROOT="$MAIN_REPO/outputs_FLAC"
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
  MEASURE_HELPER="$EXPDIR/fa_orbit_measure_worktree.sh"
  trap 'bash "$MEASURE_HELPER" --release "$SLURM_JOB_ID" "$MEASURE_ROOT" \
          || echo "could not release ${LEASE}; leaving it for the reaper" >&2' EXIT
  echo "lease OK: ${LEASE}"
fi

# --- B3. the untracked runtime assets the eval resolves RELATIVELY ------------
# A fresh worktree holds only TRACKED files, but the eval dataset config names
# "AcousticRooms" and the arm configs name "weights/AGREE/AGREE_fullAR.pt", both
# relative to the code root. Without them a pinned screen dies at startup, so
# stat every one of them here rather than 20 minutes into a queue slot.
for ASSET in AcousticRooms weights/AGREE/AGREE_fullAR.pt "$DATASET_CONFIG"; do
  [ -e "$CODE_ROOT/$ASSET" ] \
    || die "required runtime asset missing from the code root: ${CODE_ROOT}/${ASSET} — prepare the tree with fa_orbit_measure_worktree.sh - abort"
done
echo "runtime assets OK under ${CODE_ROOT} (AcousticRooms, weights/AGREE, ${DATASET_CONFIG})"
# Existence is not identity. The AGREE conditioner is part of the measured
# stack, so it is pinned by content exactly as the VAE is in fa_orbit_train.sbatch
# — a swapped-but-present file would otherwise change every metric silently.
PINNED_AGREE_SHA256="3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
AGREE_SHA="$(sha256sum "$CODE_ROOT/weights/AGREE/AGREE_fullAR.pt" | awk '{print $1}')" \
  || die "could not hash the AGREE checkpoint - abort"
[ "$AGREE_SHA" = "$PINNED_AGREE_SHA256" ] \
  || die "weights/AGREE/AGREE_fullAR.pt sha256 ${AGREE_SHA} != pinned ${PINNED_AGREE_SHA256} - abort"
echo "AGREE pin OK: ${AGREE_SHA}"

# The ViT init is lineage-relevant at eval time too: same pin the probe gates on.
python3 -c "
import sys; sys.path.insert(0, '$EXPDIR')
from fa_orbit_equiv_probe import assert_vit_pin
assert_vit_pin(); print('DINOv3 cache pin OK')" || die "DINOv3 cache pin gate FAILED - abort"

# --- C. commit binding + tracked drift ---------------------------------------
# The binding is on the CODE ROOT (the pinned worktree when one is used), so a
# main tree that keeps committing during the run is irrelevant (item 8).
HEAD_SHA="$(git -C "$CODE_ROOT" rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED - abort"
# The drift gate stays RECURSIVE over the experiment directory (so future
# subdirectory helpers, manifests and decision-relevant markdown remain covered)
# and excludes only the append-only live artifacts the running arms write:
# their Slurm transcripts and this driver's own screen logs (round-4 NIT 1).
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

# --- D2. backfill only: the checkpoint must be the AUDITED one (review B6) ----
# Parsed-config equality proves the recipe, not the file. For the historical
# lineage the plan requires an audited seed-to-file manifest, so the backfill
# checkpoint is bound to its recorded path AND sha256.
if [ "$ARM" = "C4BACKFILL" ]; then
  BACKFILL_MANIFEST="$EXPDIR/c4_backfill_manifest.json"
  [ -f "$BACKFILL_MANIFEST" ] || die "backfill manifest missing: ${BACKFILL_MANIFEST} - abort"
  python3 - "$BACKFILL_MANIFEST" "$STEP" "$CKPT" "$MAIN_REPO" <<'PY' || die "backfill lineage gate FAILED - abort"
import hashlib, json, os, sys
man, step, ckpt, main_repo = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
entry = man["checkpoints"].get(step)
if entry is None:
    sys.exit(f"step {step} is not in the audited backfill manifest "
             f"(registered: {sorted(man['checkpoints'])})")
entry_path = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(main_repo, entry["path"])
if os.path.realpath(entry_path) != os.path.realpath(ckpt):
    sys.exit(f"resolved checkpoint {ckpt} != the audited {entry_path}")
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
  # The COMMITTED registry is the only one a real run can use: FA_ORBIT_ARM_REGISTRY
  # is honoured solely for a synthetic OUTPUT_ROOT, and a SLURM run has already
  # been forced onto the production output root above — so this is a test seam,
  # not a bypass.
  ARM_REGISTRY="$EXPDIR/arm_launch_registry.json"
  if [ -n "${FA_ORBIT_ARM_REGISTRY:-}" ]; then
    [ "$OUTPUT_ROOT" != "$PRODUCTION_OUTPUT_ROOT" ] \
      || die "FA_ORBIT_ARM_REGISTRY may not override the audited registry for the production output root - abort"
    ARM_REGISTRY="$FA_ORBIT_ARM_REGISTRY"
  fi
  [ -f "$ARM_LAUNCH_MANIFEST" ] || die "arm launch manifest missing: ${ARM_LAUNCH_MANIFEST} — a screen may only evaluate a checkpoint from a recorded launch - abort"
  [ -f "$ARM_REGISTRY" ] || die "audited arm launch registry missing: ${ARM_REGISTRY} - abort"
  python3 - "$ARM_LAUNCH_MANIFEST" "$ARM" "$CKPT" "$CONFIG_SHA" "$CKPT_DIR" "$MAIN_REPO" "$ARM_REGISTRY" <<'PY' || die "arm launch-manifest lineage gate FAILED - abort"
import hashlib, json, os, sys
man_path, arm, ckpt, cfg_sha, ckpt_dir, main_repo, reg_path = sys.argv[1:8]
man = {}
for line in open(man_path):
    line = line.strip()
    if line and not line.startswith("#"):
        k, _, rest = line.partition(" ")
        man[k] = rest.strip()
fields = ("arm " + man.get("arm", "")).split()
kv = {fields[i]: fields[i + 1] for i in range(0, len(fields) - 1, 2)}
jf = ("job " + man.get("job", "")).split()
jkv = {jf[i]: jf[i + 1] for i in range(0, len(jf) - 1, 2)}
bad = []
# The manifest lives under gitignored outputs_FLAC and is MUTABLE: binding to
# "whatever it says now" proves nothing, because anything that could forge a
# checkpoint could forge its manifest too. Bind it to the COMMITTED, audited
# registry — same bytes, and every launch fact re-checked field by field.
reg = json.load(open(reg_path))["arms"].get(arm)
if reg is None:
    sys.exit(f"ARM LINEAGE GATE: {arm} is not in the audited launch registry {reg_path}")
got_sha = hashlib.sha256(open(man_path, "rb").read()).hexdigest()
if got_sha != reg["manifest_sha256"]:
    bad.append(f"launch manifest sha256 {got_sha[:12]} != audited {reg['manifest_sha256'][:12]} — "
               "the manifest changed after it was registered")
for label, got_v, want_v in (("job", jkv.get("job"), reg["job"]),
                             ("launch mode", jkv.get("mode"), reg["mode"]),
                             ("launch_uuid", jkv.get("launch_uuid"), reg["launch_uuid"]),
                             ("commit", man.get("commit"), reg["commit"]),
                             ("rung", kv.get("rung"), reg["rung"]),
                             ("max_steps", kv.get("max_steps"), reg["max_steps"]),
                             ("config_sha256", man.get("config_sha256"), reg["config_sha256"]),
                             ("vae_sha256", man.get("vae_sha256"), reg["vae_sha256"]),
                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"), reg["p0_manifest_sha256"]),
                             ("save_dir", man.get("save_dir"), reg["save_dir"])):
    if got_v != want_v:
        bad.append(f"{label} {got_v!r} != registered {want_v!r}")
if jkv.get("mode") != "INITIAL":
    bad.append(f"launch mode {jkv.get('mode')!r} is not INITIAL (a restart is not a registered launch)")
if int(reg.get("training_seed", -1)) != 42:
    bad.append(f"registered training seed {reg.get('training_seed')!r} != 42")
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
    # save_dir is recorded relative to the MAIN tree; under worktree-pinned
    # execution the cwd is the pinned code root, so resolve it explicitly.
    save_abs = save_dir if os.path.isabs(save_dir) else os.path.join(main_repo, save_dir)
    canon = os.path.realpath(os.path.join(save_abs, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))
    if want_dir != canon:
        bad.append(f"checkpoint dir {want_dir} is not the manifest's canonical {canon}")
    if os.path.realpath(os.path.dirname(ckpt)) != canon:
        bad.append(f"checkpoint {ckpt} does not live in the canonical run directory {canon}")
if bad:
    sys.exit("ARM LINEAGE GATE: " + "; ".join(bad))
print(f"arm lineage OK: {arm} bound to AUDITED launch job {reg['job']} ({reg['mode']}, uuid "
      f"{reg['launch_uuid'][:8]}, commit {reg['commit'][:12]}, rung {reg['rung']}, "
      f"max_steps {reg['max_steps']}, training seed {reg['training_seed']}); manifest "
      f"{got_sha[:12]} as registered; config {cfg_sha[:12]}")
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
LOG="${LOG:-${LOGDIR}/fa_orbit_${TS}_${ARM}_S${STEP}_s${SEED}_K${K}_screen.log}"
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
# Recorded repo-RELATIVE: an absolute path into a pinned worktree becomes a
# dangling reference the moment that (prunable) tree is removed.
MODEL_CONFIG_REL="$(python3 -c "import os,sys; print(os.path.relpath(os.path.realpath(sys.argv[1]), sys.argv[2]))" "$MODEL_CONFIG" "$CODE_ROOT")" \
  || die "could not make the model-config path repo-relative - abort"
python3 - "$METRICS" "$ARM" "$STEP" "$SEED" "$K" "$EVAL_NAME" "$MODEL_CONFIG_REL" "$CONFIG_SHA" \
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
    --contract futility --verify-hashes --output-root "$MAIN_REPO" "$METRICS" \
    || die "row validation FAILED - refusing to emit SCREENRESULT for an unprovable row" 6
  VALIDATED="futility"
else
  VALIDATED="pending-cell"
  echo "NOTE: CELL=${CELL} is one row of a five-seed table cell; it is validated as a WHOLE cell"
  echo "  once every seed has landed, and until then this row is NOT validated:"
  echo "  python3 ${EXPDIR}/exp11_validate_rows.py --arm ${ARM} --step ${STEP} --k ${K} --contract table --verify-hashes --output-root ${MAIN_REPO} <all seed JSONs>"
fi

echo "SCREENRESULT arm=${ARM} cell=${CELL} step=${STEP} seed=${SEED} k=${K} rc=${EVAL_RC} validated=${VALIDATED} eval_name=${EVAL_NAME} metrics=${METRICS} sidecar=${METRICS}.screenmeta.json config_sha256=${CONFIG_SHA} ckpt_sha256=${CKPT_SHA} commit=${HEAD_SHA} log=${LOG}"
# --- END screen-run-and-record -------------------------------------------------
exit 0
