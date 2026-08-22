#!/usr/bin/env bash
# ============================================================================
# bfc_launch_guardtests.sh - guard-branch exercise for bfc_launch.sh
# (same shell-pragmatic form as exp_13's dtail_launch_guardtests.sh).
#
# Drives the REAL launcher through every fail-closed branch AND through a
# complete valid launch, without ever training. Two mechanisms:
#   * DRY_RUN=1 - the launcher's own rehearsal mode: every gate executes, the
#     assembled command is printed as a `LAUNCH-CMD:` line, train.py is not
#     executed. This is what the flag pins below read.
#   * dry-fail stops for the branches that must abort BEFORE the expensive
#     gates: MIN_FREE_DISK_MB=99999999 (the df floor, which sits before the VRAM
#     gate, the wandb gate and the init-identity audit) and MIN_FREE_MB=99000000
#     (the VRAM gate, one step later).
# No training is launched and no GPU work is done; the init-identity audit runs
# on CPU with CUDA_VISIBLE_DEVICES="" inside the launcher.
#
# Safety rules honoured here:
#   * the arm config ${EXPDIR21}/FLAC_AR_BFC.json is temporarily swapped by the
#     contract cases. It is backed up first, restored immediately after each
#     case AND from the EXIT trap, restoration is verified, and the original
#     sha256 is echoed into this log so it can be recovered by hand if the
#     script is hard-killed mid-case;
#   * nothing under outputs_FLAC/ is created, read or removed - the launcher has
#     no mkdir at all, and the exercise asserts the arm's save-dir is not
#     conjured into existence;
#   * launcher logs created by this exercise are removed at the end (the
#     launcher tees a timestamped *_train.log per invocation).
#
# Usage (env flac must be active):
#   bash worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_launch_guardtests.sh
# Exit 0 = all cases behaved as specified.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR21="worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude"
LAUNCHER="${EXPDIR21}/bfc_launch.sh"
BFC_CONFIG="${EXPDIR21}/FLAC_AR_BFC.json"
BF_CONFIG="worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json"
SAVEDIR="outputs_FLAC/exp21_BFC"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR21}/bf_fa_cartesian_${TS}_guardtests.log"
DRYFAIL_MIN_FREE=99000000          # forces the per-GPU VRAM gate to abort
DRYFAIL_MIN_DISK=99999999          # forces the df floor to abort (earlier stop)

exec > >(tee -a "$LOG") 2>&1
echo "=== bfc_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ]   || { echo "launcher not found - abort"; exit 3; }
[ -f "$BFC_CONFIG" ] || { echo "arm config not found: ${BFC_CONFIG} - abort"; exit 3; }
[ -f "$BF_CONFIG" ]  || { echo "B-F reference config not found: ${BF_CONFIG} - abort"; exit 3; }

SAVEDIR_EXISTED=0; [ -e "$SAVEDIR" ] && SAVEDIR_EXISTED=1

TMP="$(mktemp -d)"
CFG_BACKUP="${TMP}/FLAC_AR_BFC.json.orig"
cp "$BFC_CONFIG" "$CFG_BACKUP"
CFG_ORIG_SHA="$(sha256sum "$BFC_CONFIG" | awk '{print $1}')"
echo "arm config sha256 (manual recovery if hard-killed): ${CFG_ORIG_SHA}"

# the launcher's own per-invocation logs (*_train.log for a real launch,
# *_dryrun.log for a rehearsal). This exercise's own *_guardtests.log is
# deliberately NOT in the pattern: it is the evidence this run leaves behind.
LOG_PAT='_(train|dryrun)\.log$'
LOGS_BEFORE="$(ls "${EXPDIR21}" | grep -E "$LOG_PAT" | sort || true)"

restore_cfg() {
  [ -f "$CFG_BACKUP" ] || return 0
  cp -f "$CFG_BACKUP" "$BFC_CONFIG"
  if cmp -s "$CFG_BACKUP" "$BFC_CONFIG"; then
    echo "  arm config restored OK (sha256 ${CFG_ORIG_SHA})"
  else
    echo "  !!! ARM CONFIG NOT RESTORED - restore ${BFC_CONFIG} from git (sha256 ${CFG_ORIG_SHA})"
  fi
}

cleanup() {
  restore_cfg
  rm -rf "$TMP"
  local now removed
  now="$(ls "${EXPDIR21}" | grep -E "$LOG_PAT" | sort || true)"
  removed="$(comm -13 <(echo "$LOGS_BEFORE") <(echo "$now"))"
  if [ -n "$removed" ]; then
    echo "--- removing launcher logs created by this exercise ---"
    echo "$removed" | while read -r f; do [ -n "$f" ] && rm -f "${EXPDIR21}/${f}" && echo "  rm ${f}"; done
  fi
}
trap cleanup EXIT

# --- a mutation helper: rewrite the arm config through python, never by hand ---
mutate_cfg() { BFC_CONFIG="$BFC_CONFIG" python3 -c "$1"; }

PASS=0; FAIL=0
# case <label> <want_rc> <want_substrings, '|||'-separated> -- <ENV=VAL>...
case_run() {
  local label="$1" want_rc="$2" want_msgs="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local out rc ok=1 m rest missing=""
  out="$(env "$@" bash "$LAUNCHER" 2>&1)"; rc=$?
  [ "$rc" = "$want_rc" ] || ok=0
  rest="$want_msgs"
  while [ -n "$rest" ]; do
    m="${rest%%|||*}"
    if [ "$rest" = "$m" ]; then rest=""; else rest="${rest#*|||}"; fi
    case "$out" in *"$m"*) ;; *) ok=0; missing="${missing}
      missing substring: ${m}";; esac
  done
  if [ "$ok" = "1" ]; then
    PASS=$((PASS+1)); printf 'PASS  rc=%-3s %s\n' "$rc" "$label"
  else
    FAIL=$((FAIL+1))
    printf 'FAIL  rc=%s (want %s) %s%s\n' "$rc" "$want_rc" "$label" "$missing"
    echo "$out" | sed 's/^/      | /' | tail -14
  fi
}

check() { # check <label> <condition-rc> ; a plain boolean assertion
  local label="$1" rc="$2"
  if [ "$rc" = "0" ]; then PASS=$((PASS+1)); echo "PASS  ${label}"
  else FAIL=$((FAIL+1)); echo "FAIL  ${label}"; fi
}

echo "--- 0. static syntax ---"
bash -n "$LAUNCHER"; check "bash -n ${LAUNCHER}" $?
bash -n "${EXPDIR21}/bfc_launch_guardtests.sh"; check "bash -n bfc_launch_guardtests.sh" $?

echo "--- A. environment / knob validation ---"
case_run "wrong conda env rejected" 2 "CONDA_DEFAULT_ENV must be 'flac'" -- CONDA_DEFAULT_ENV=rir2rir
case_run "CHECKPOINT_EVERY=0 rejected" 2 "CHECKPOINT_EVERY must be > 0" -- CHECKPOINT_EVERY=0
case_run "CHECKPOINT_EVERY=abc rejected" 2 "CHECKPOINT_EVERY must be a positive integer" -- CHECKPOINT_EVERY=abc
case_run "MAXSTEPS=abc rejected" 2 "MAXSTEPS must be a positive integer" -- MAXSTEPS=abc
case_run "MAXSTEPS=0 rejected" 2 "MAXSTEPS must be > 0" -- MAXSTEPS=0
case_run "MAXSTEPS=40000.5 (non-integer) rejected" 2 "MAXSTEPS must be a positive integer" -- MAXSTEPS=40000.5
case_run "MB=8 ACC=8 rung rejected (BN never sees accumulation)" 2 \
  "only the BN-compliant rung MB=32 ACC=1" -- MB=8 ACC=8
case_run "MB=64 ACC=1 rung rejected (BN=128, not the paper's 64)" 2 \
  "only the BN-compliant rung MB=32 ACC=1" -- MB=64
case_run "MIN_FREE_DISK_MB=abc rejected" 2 "MIN_FREE_DISK_MB must be a positive integer" -- MIN_FREE_DISK_MB=abc
case_run "MIN_FREE_MB=abc rejected" 2 "MIN_FREE_MB must be a positive integer" \
  -- MIN_FREE_MB=abc MIN_FREE_DISK_MB=1

echo "--- B. config contract (arm config temporarily swapped; restored after each case) ---"
# B1: the method itself reverted to B-F's
mutate_cfg 'import json,os;p=os.environ["BFC_CONFIG"];c=json.load(open(p));c["training"]["cond_method"]="fa_invariant";json.dump(c,open(p,"w"),indent=4)'
case_run "B1 cond_method reverted to fa_invariant rejected" 2 \
  "not 'fa_cartesian'|||config contract FAILED" -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# B2: the DECLARED training chunk cap changed (announcement 06: a different cap
# at this rung is a different method, not a tuning knob)
mutate_cfg 'import json,os;p=os.environ["BFC_CONFIG"];c=json.load(open(p));c["training"]["frame_avg_max_fwd_samples"]=64;json.dump(c,open(p,"w"),indent=4)'
case_run "B2 training cap 64 (not the declared 32) rejected" 2 \
  "not the declared 32|||config contract FAILED" -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# B3: a NON-declared drift from B-F - the arm would silently be two mechanisms
mutate_cfg 'import json,os;p=os.environ["BFC_CONFIG"];c=json.load(open(p));c["training"]["cfg_dropout_prob"]=0.2;json.dump(c,open(p,"w"),indent=4)'
case_run "B3 non-declared drift from B-F rejected" 2 \
  "not a single-delta change|||config contract FAILED" -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# B4: ViT gradient checkpointing dropped (the B-F recipe enables it via CONFIG,
# so nothing on the command line would reveal its absence)
mutate_cfg 'import json,os;p=os.environ["BFC_CONFIG"];c=json.load(open(p));[cc["config"].pop("gradient_checkpointing",None) for cc in c["model"]["conditioning"]["configs"] if cc.get("id")=="source_vit"];json.dump(c,open(p,"w"),indent=4)'
case_run "B4 ViT gradient_checkpointing dropped on one ViT rejected" 2 \
  "gradient_checkpointing on BOTH ViT conditioners, found 1|||config contract FAILED" \
  -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# B5: yaw augmentation smuggled in beside a frame-averaged method
mutate_cfg 'import json,os;p=os.environ["BFC_CONFIG"];c=json.load(open(p));c["training"]["yaw_aug"]={"mode":"random"};json.dump(c,open(p,"w"),indent=4)'
case_run "B5 yaw_aug beside a frame-averaged method rejected" 2 \
  "mutually exclusive|||config contract FAILED" -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

# B6: the C4 orbit itself changed
mutate_cfg 'import json,os;p=os.environ["BFC_CONFIG"];c=json.load(open(p));c["training"]["frame_avg_angles"]=[0.0,180.0];json.dump(c,open(p,"w"),indent=4)'
case_run "B6 non-C4 frame_avg_angles rejected" 2 \
  "is not the C4 orbit|||config contract FAILED" -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
restore_cfg

echo "--- C. df floor on the outputs volume (bypass for guard-testing: MIN_FREE_DISK_MB=1) ---"
case_run "C1 df floor rejects an impossible free-space requirement" 2 \
  "config contract OK|||free disk|||refusing to launch" -- MIN_FREE_DISK_MB="$DRYFAIL_MIN_DISK"
case_run "C2 df bypass reaches the per-GPU VRAM gate" 2 \
  "MiB free on the volume|||refusing to launch" \
  -- MIN_FREE_DISK_MB=1 MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- D. per-GPU free-VRAM floor (a co-tenancy FLOOR, not exclusivity) ---"
case_run "D1 an unsatisfiable VRAM floor refuses to launch" 2 \
  "GPU 0 free|||< required 99000000 MiB|||refusing to launch" -- MIN_FREE_MB="$DRYFAIL_MIN_FREE"

echo "--- E. the FULL dry run: every gate, no training (LOGGER=none) ---"
DRY_OUT="$(env DRY_RUN=1 LOGGER=none bash "$LAUNCHER" 2>&1)"; DRY_RC=$?
check "E1 full dry run exits 0 (rc=${DRY_RC})" "$([ "$DRY_RC" = "0" ] && echo 0 || echo 1)"
if [ "$DRY_RC" != "0" ]; then
  echo "$DRY_OUT" | sed 's/^/      | /' | tail -25
fi
for want in \
  "env gate: python" \
  "config contract OK: BFC == BF + cond_method 'fa_cartesian' + training cap 32" \
  "chunk plan: cap 32 / micro-batch 32 / C4 -> angles_per_chunk 1" \
  "ViT gradient_checkpointing: True on both ViT conditioners" \
  "MiB free on the volume" \
  "co-tenancy disclosure" \
  "ViT pin OK" \
  "wiring: BFC cond_method='fa_cartesian'" \
  "architecture: identical param names" \
  "init identity: state_dict sha256 match under seed 42" \
  "pip-freeze sha256:" \
  "LAUNCH-CMD:" \
  "DRY RUN: every gate above ran"; do
  case "$DRY_OUT" in *"$want"*) check "E2 dry run reports: ${want}" 0;; *) check "E2 dry run reports: ${want}" 1;; esac
done
# ...and it must NOT have trained, nor consulted wandb under LOGGER=none
case "$DRY_OUT" in *"BFC training exited rc="*) check "E3 train.py was NOT executed" 1;; *) check "E3 train.py was NOT executed" 0;; esac
case "$DRY_OUT" in *"wandb identity"*) check "E4 LOGGER=none skips the wandb gate" 1;; *) check "E4 LOGGER=none skips the wandb gate" 0;; esac

echo "--- F. the assembled command, flag by flag (the LAUNCH-CMD line) ---"
CMDLINE="$(printf '%s\n' "$DRY_OUT" | grep -m1 '^LAUNCH-CMD: ')"
check "F0 a LAUNCH-CMD line was printed" "$([ -n "$CMDLINE" ] && echo 0 || echo 1)"
echo "  ${CMDLINE}"
for want in \
  "python train.py" \
  "HF_HUB_OFFLINE=1" \
  "CUDA_VISIBLE_DEVICES=0,1" \
  "--model-config worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json" \
  "--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json" \
  "--pretransform-ckpt-path weights/FLAC/VAE.safetensors" \
  "--max-steps 40000" \
  "--batch-size 32" \
  "--accum-batches 1" \
  "--num-workers 6" \
  "--seed 42" \
  "--num-gpus 2" \
  "--strategy ddp_find_unused_parameters_true" \
  "--sync-batchnorm true" \
  "--precision bf16-mixed" \
  "--checkpoint-every 2500" \
  "--name FLAC_exp21_BFC" \
  "--experiment-name exp21_BFC" \
  "--save-dir outputs_FLAC/exp21_BFC"; do
  case "$CMDLINE" in *"$want"*) check "F1 command carries: ${want}" 0;; *) check "F1 command carries: ${want}" 1;; esac
done
# The absences are the point of half this file. --val-dataset-config would break
# single-delta parity (B-F ran without a validation loader; a validation pass
# draws RNG noise). A resume flag would make this not a from-scratch run.
for bad in "--val-dataset-config" " --ckpt-path " "--pretrained-ckpt-path" "--recover"; do
  case "$CMDLINE" in *"$bad"*) check "F2 command does NOT carry: ${bad}" 1;; *) check "F2 command does NOT carry: ${bad}" 0;; esac
done
# ...and no validation config anywhere in the whole dry run, under any spelling
case "$DRY_OUT" in *"acousticroom_seeneval"*|*"unseeneval"*) check "F3 no eval dataset config anywhere in the run" 1;;
  *) check "F3 no eval dataset config anywhere in the run" 0;; esac

echo "--- G. the wandb identity gate runs under the registered logger ---"
WB_OUT="$(env DRY_RUN=1 LOGGER=wandb bash "$LAUNCHER" 2>&1)"; WB_RC=$?
check "G1 full dry run under LOGGER=wandb exits 0 (rc=${WB_RC})" "$([ "$WB_RC" = "0" ] && echo 0 || echo 1)"
case "$WB_OUT" in *"wandb identity: yh4742@princeton.edu"*) check "G2 the wandb identity gate ran and matched" 0;;
  *) check "G2 the wandb identity gate ran and matched" 1; echo "$WB_OUT" | grep -i wandb | sed 's/^/      | /' | tail -5;; esac
case "$WB_OUT" in *"--logger wandb"*) check "G3 the assembled command carries --logger wandb" 0;;
  *) check "G3 the assembled command carries --logger wandb" 1;; esac

echo "--- H. the exercise created nothing under outputs_FLAC/ ---"
if [ "$SAVEDIR_EXISTED" = "1" ]; then
  echo "SKIP  ${SAVEDIR} already existed before this exercise - not asserting its absence"
else
  check "H1 ${SAVEDIR} was not created by any gate" "$([ -e "$SAVEDIR" ] && echo 1 || echo 0)"
fi

echo "--- arm-config integrity re-check ---"
NOW_CFG_SHA="$(sha256sum "$BFC_CONFIG" | awk '{print $1}')"
if [ "$NOW_CFG_SHA" = "$CFG_ORIG_SHA" ]; then
  PASS=$((PASS+1)); echo "PASS  arm config restored (${NOW_CFG_SHA})"
else
  FAIL=$((FAIL+1)); echo "FAIL  ARM CONFIG NOT RESTORED: ${NOW_CFG_SHA} != ${CFG_ORIG_SHA}"
fi

echo
echo "=== guard exercise: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
exit 0
