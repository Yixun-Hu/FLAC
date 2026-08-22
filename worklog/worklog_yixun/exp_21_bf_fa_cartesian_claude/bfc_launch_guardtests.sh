#!/usr/bin/env bash
# ============================================================================
# bfc_launch_guardtests.sh - guard-branch exercise for bfc_launch.sh
# (same shell-pragmatic form as exp_13's dtail_launch_guardtests.sh).
#
# Drives the REAL launcher through every fail-closed branch AND through both of
# its valid modes, without ever training. Three mechanisms:
#   * DRY_RUN=1 - the launcher's own rehearsal mode: every gate executes, the
#     assembled argv is printed (as a human `LAUNCH-CMD:` line AND as a
#     one-token-per-line LAUNCH-ARGV block), train.py is not executed. The
#     manifest comparison below reads the token block, not the line: a substring
#     check cannot see a token that was split, duplicated, reordered or appended.
#   * SMOKE=1 - the launcher's sanctioned short mode, checked to be a DIFFERENT
#     manifest in a DIFFERENT namespace, never a loosened registered one.
#   * dry-fail stops for the branches that must abort BEFORE the expensive
#     gates: MIN_FREE_DISK_MB=99999999 (the df floor, which sits before the VRAM
#     gate, the wandb gate and the init-identity audit) and MIN_FREE_MB=99000000
#     (the VRAM gate, one step later).
#
# ⚠ THE RULE THIS FILE LEARNED THE HARD WAY (round-5 red phase): a case that
# expects a REJECTION must be unable to train even when the rejection is absent.
# Section D2's first run, against the unfixed launcher, reached a real training
# start and created a wandb run before it was killed - the very bypass it exists
# to forbid. Every rejection case therefore also carries a dry-fail stop, so a
# missing guard fails the case on a missing message instead of launching a run.
#
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
#     launcher tees one timestamped log per invocation: *_train.log for a real
#     launch, *_smoke.log for a smoke run, *_dryrun.log for a rehearsal - and
#     section G6 asserts a rehearsal produces only the last of those).
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
SMOKE_SAVEDIR="outputs_FLAC/exp21_BFC_smoke"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR21}/bf_fa_cartesian_${TS}_guardtests.log"
# The launcher PINS its two resource floors in the registered run (r5 review
# nit): an override there aborts. This exercise exists to drive those gates to
# their failure branches, so it declares itself -- and the cases below prove the
# pin still holds for anyone who has not.
export GUARDTEST=1
DRYFAIL_MIN_FREE=99000000          # forces the per-GPU VRAM gate to abort
DRYFAIL_MIN_DISK=99999999          # forces the df floor to abort (earlier stop)

exec > >(tee -a "$LOG") 2>&1
echo "=== bfc_launch guard exercise - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) ==="
echo "launcher: ${LAUNCHER}"

[ -f "$LAUNCHER" ]   || { echo "launcher not found - abort"; exit 3; }
[ -f "$BFC_CONFIG" ] || { echo "arm config not found: ${BFC_CONFIG} - abort"; exit 3; }
[ -f "$BF_CONFIG" ]  || { echo "B-F reference config not found: ${BF_CONFIG} - abort"; exit 3; }

SAVEDIR_EXISTED=0; [ -e "$SAVEDIR" ] && SAVEDIR_EXISTED=1
SMOKE_SAVEDIR_EXISTED=0; [ -e "$SMOKE_SAVEDIR" ] && SMOKE_SAVEDIR_EXISTED=1

TMP="$(mktemp -d)"
CFG_BACKUP="${TMP}/FLAC_AR_BFC.json.orig"
cp "$BFC_CONFIG" "$CFG_BACKUP"
CFG_ORIG_SHA="$(sha256sum "$BFC_CONFIG" | awk '{print $1}')"
echo "arm config sha256 (manual recovery if hard-killed): ${CFG_ORIG_SHA}"

# the launcher's own per-invocation logs (*_train.log for a real launch,
# *_smoke.log for the sanctioned short run, *_dryrun.log for a rehearsal). This
# exercise's own *_guardtests.log is deliberately NOT in the pattern: it is the
# evidence this run leaves behind.
LOG_PAT='_(train|smoke|dryrun)\.log$'
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

echo "--- D3. the two resource FLOORS are pinned in the registered run (r5 nit) ---"
# The floors were env-overridable in every mode, so a registered launch could be
# lowered onto a GPU or volume that cannot hold it -- the r4 MAXSTEPS/LOGGER
# defect, one gate over. Outside a rehearsal the override now ABORTS; these cases
# run WITHOUT this exercise's GUARDTEST declaration, which is what the rest of the
# file relies on, so they prove the pin holds for anyone who has not declared it.
case_run "D3a MIN_FREE_MB override refused in the registered run" 2 \
  "may not be overridden in the REGISTERED run|||resource" -- -u GUARDTEST MIN_FREE_MB=1
case_run "D3b MIN_FREE_DISK_MB override refused in the registered run" 2 \
  "may not be overridden in the REGISTERED run" -- -u GUARDTEST MIN_FREE_DISK_MB=1
case_run "D3c a rehearsal may still lower the floors (DRY_RUN)" 0 \
  "DRY RUN" -- -u GUARDTEST DRY_RUN=1 MIN_FREE_MB=1 MIN_FREE_DISK_MB=1
case_run "D3d SMOKE may still lower the floors" 0 \
  "DRY RUN (SMOKE)" -- -u GUARDTEST DRY_RUN=1 SMOKE=1 MIN_FREE_MB=1 MIN_FREE_DISK_MB=1
# NOTE deliberately absent: a "registered run with NO override" case. Without an
# override there is nothing to dry-fail on, so if the box happened to satisfy both
# real floors the case would run to a REAL train.py start -- the round-4 incident
# exactly. The no-override path is covered by the full DRY_RUN rehearsal below,
# which stops itself.

echo "--- D2. the REGISTERED MANIFEST is not overridable (r4 review BLOCKING 1) ---"
# Every one of these used to pass every gate and train an unapproved recipe.
#
# ⚠ EVERY case here carries the df dry-fail stop, and that is a SAFETY property,
# not tidiness. A rejection case whose rejection is missing must still abort
# before train.py: the first red run of this section (against the unfixed
# launcher) reached a REAL training start and created a wandb run before it was
# killed. With MIN_FREE_DISK_MB pinned impossibly high, a missing manifest pin
# now fails the case on the absent substring instead of launching the recipe the
# case exists to forbid. The cases that set DRY_RUN=1 are safe by that alone,
# but they carry the stop too, so no case in this section depends on which guard
# happens to fire.
DF_STOP="MIN_FREE_DISK_MB=${DRYFAIL_MIN_DISK}"
case_run "D2a MAXSTEPS=50000 rejected" 2 \
  "MAXSTEPS|||40000|||SMOKE=1" -- MAXSTEPS=50000 "$DF_STOP"
case_run "D2b MAXSTEPS=39999 rejected (shorter is also a different recipe)" 2 \
  "MAXSTEPS|||40000|||SMOKE=1" -- MAXSTEPS=39999 "$DF_STOP"
case_run "D2c CHECKPOINT_EVERY=1 rejected" 2 \
  "CHECKPOINT_EVERY|||2500|||SMOKE=1" -- CHECKPOINT_EVERY=1 "$DF_STOP"
case_run "D2d CHECKPOINT_EVERY=1250 rejected" 2 \
  "CHECKPOINT_EVERY|||2500|||SMOKE=1" -- CHECKPOINT_EVERY=1250 "$DF_STOP"
case_run "D2e LOGGER=none rejected in the registered mode (B-F ran on wandb)" 2 \
  "LOGGER|||wandb|||SMOKE=1" -- LOGGER=none "$DF_STOP"
case_run "D2f LOGGER=comet rejected" 2 "LOGGER|||wandb|||SMOKE=1" -- LOGGER=comet "$DF_STOP"
case_run "D2g DRY_RUN=2 rejected (fail closed, not 'not 1, therefore train')" 2 \
  "DRY_RUN must be 0 or 1" -- DRY_RUN=2 "$DF_STOP"
case_run "D2h DRY_RUN=yes rejected" 2 "DRY_RUN must be 0 or 1" -- DRY_RUN=yes "$DF_STOP"
case_run "D2i SMOKE=2 rejected" 2 "SMOKE must be 0 or 1" -- SMOKE=2 "$DF_STOP"
case_run "D2j SMOKE=1 with an over-cap MAXSTEPS rejected" 2 \
  "smoke|||50" -- SMOKE=1 MAXSTEPS=5000 DRY_RUN=1 "$DF_STOP"
case_run "D2k SMOKE=1 LOGGER=wandb rejected (a smoke run never touches the project)" 2 \
  "LOGGER" -- SMOKE=1 LOGGER=wandb DRY_RUN=1 "$DF_STOP"

echo "--- E. the FULL dry run: every gate, the REGISTERED manifest ---"
DRY_OUT="$(env DRY_RUN=1 bash "$LAUNCHER" 2>&1)"; DRY_RC=$?
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
  "DRY RUN (REGISTERED): every gate above ran"; do
  case "$DRY_OUT" in *"$want"*) check "E2 dry run reports: ${want}" 0;; *) check "E2 dry run reports: ${want}" 1;; esac
done
# ...and it must NOT have trained, while the REGISTERED logger's identity gate must run
case "$DRY_OUT" in *"BFC training exited rc="*) check "E3 train.py was NOT executed" 1;; *) check "E3 train.py was NOT executed" 0;; esac
case "$DRY_OUT" in *"wandb identity: yh4742@princeton.edu"*) check "E4 the wandb identity gate ran and matched" 0;;
  *) check "E4 the wandb identity gate ran and matched" 1; echo "$DRY_OUT" | grep -i wandb | sed 's/^/      | /' | tail -5;; esac

echo "--- F. the assembled argv, TOKEN BY TOKEN (r4 nit: exact vector, not substrings) ---"
# A substring check cannot see a token that was split, duplicated, reordered or
# appended after the ones it looked for. The launcher emits its argv one token
# per line between markers; this compares that vector element-for-element with
# the approved manifest, and fails on the FIRST difference.
argv_of() { printf '%s\n' "$1" | awk '/^LAUNCH-ARGV-BEGIN$/{f=1;next} /^LAUNCH-ARGV-END$/{f=0} f'; }

REG_ARGV=(env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py
  --model-config worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps 40000 --batch-size 32 --accum-batches 1
  --num-workers 6 --seed 42
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --precision bf16-mixed
  --logger wandb --checkpoint-every 2500
  --name FLAC_exp21_BFC --experiment-name exp21_BFC
  --save-dir outputs_FLAC/exp21_BFC)

compare_argv() { # $1=label $2=output ; then the expected vector as "$@"
  local label="$1" out="$2"; shift 2
  local -a want=("$@") got=()
  local line
  while IFS= read -r line; do got+=("$line"); done < <(argv_of "$out")
  if [ "${#got[@]}" -eq 0 ]; then
    FAIL=$((FAIL+1)); echo "FAIL  ${label}: no LAUNCH-ARGV block was emitted"; return
  fi
  if [ "${#got[@]}" -ne "${#want[@]}" ]; then
    FAIL=$((FAIL+1))
    echo "FAIL  ${label}: argv has ${#got[@]} tokens, the approved manifest has ${#want[@]}"
    printf '      | got:  %s\n' "${got[*]}"
    printf '      | want: %s\n' "${want[*]}"
    return
  fi
  local i
  for i in "${!want[@]}"; do
    if [ "${got[$i]}" != "${want[$i]}" ]; then
      FAIL=$((FAIL+1))
      echo "FAIL  ${label}: argv token ${i} is '${got[$i]}', the approved manifest says '${want[$i]}'"
      return
    fi
  done
  PASS=$((PASS+1)); echo "PASS  ${label} (${#got[@]} tokens, exact)"
}

compare_argv "F1 registered argv matches the approved manifest token for token" \
  "$DRY_OUT" "${REG_ARGV[@]}"
# The absences are the point of half this file. --val-dataset-config would break
# single-delta parity (B-F ran without a validation loader; a validation pass
# draws RNG noise). A resume flag would make this not a from-scratch run. The
# exact-vector check above already proves they are absent; these name them, so a
# future manifest edit that adds one fails with a message that says which.
CMDLINE="$(printf '%s\n' "$DRY_OUT" | grep -m1 '^LAUNCH-CMD: ')"
check "F0 a LAUNCH-CMD line was printed for humans" "$([ -n "$CMDLINE" ] && echo 0 || echo 1)"
echo "  ${CMDLINE}"
for bad in "--val-dataset-config" " --ckpt-path " "--pretrained-ckpt-path" "--recover"; do
  case "$CMDLINE" in *"$bad"*) check "F2 command does NOT carry: ${bad}" 1;; *) check "F2 command does NOT carry: ${bad}" 0;; esac
done
# ...and no validation config anywhere in the whole dry run, under any spelling
case "$DRY_OUT" in *"acousticroom_seeneval"*|*"unseeneval"*) check "F3 no eval dataset config anywhere in the run" 1;;
  *) check "F3 no eval dataset config anywhere in the run" 0;; esac

echo "--- G. SMOKE mode: the ONLY sanctioned short run, in its own namespace ---"
SMOKE_OUT="$(env DRY_RUN=1 SMOKE=1 bash "$LAUNCHER" 2>&1)"; SMOKE_RC=$?
check "G1 smoke dry run exits 0 (rc=${SMOKE_RC})" "$([ "$SMOKE_RC" = "0" ] && echo 0 || echo 1)"
if [ "$SMOKE_RC" != "0" ]; then echo "$SMOKE_OUT" | sed 's/^/      | /' | tail -20; fi
SMOKE_ARGV=(env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py
  --model-config worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/FLAC_AR_BFC.json
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors
  --max-steps 25 --batch-size 32 --accum-batches 1
  --num-workers 6 --seed 42
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
  --precision bf16-mixed
  --logger none --checkpoint-every 1000025
  --name FLAC_exp21_BFC_smoke --experiment-name exp21_BFC_smoke
  --save-dir outputs_FLAC/exp21_BFC_smoke)
compare_argv "G2 smoke argv is the smoke manifest, token for token" \
  "$SMOKE_OUT" "${SMOKE_ARGV[@]}"
case "$SMOKE_OUT" in *"wandb identity"*) check "G3 a smoke run never consults wandb" 1;; *) check "G3 a smoke run never consults wandb" 0;; esac
case "$SMOKE_OUT" in *"SMOKE"*) check "G4 the smoke banner says so" 0;; *) check "G4 the smoke banner says so" 1;; esac
# the registered namespace must be unreachable from a smoke run
case "$SMOKE_OUT" in *"--save-dir outputs_FLAC/exp21_BFC "*|*"--save-dir outputs_FLAC/exp21_BFC") check "G5 smoke cannot write into the registered save-dir" 1;;
  *) check "G5 smoke cannot write into the registered save-dir" 0;; esac

echo "--- G6. a dry run leaves a _dryrun.log and NOTHING that reads like a training log ---"
BEFORE_LOGS="$(ls "${EXPDIR21}" | grep -E "$LOG_PAT" | sort || true)"
env DRY_RUN=1 bash "$LAUNCHER" > /dev/null 2>&1
AFTER_LOGS="$(ls "${EXPDIR21}" | grep -E "$LOG_PAT" | sort || true)"
NEW_LOGS="$(comm -13 <(echo "$BEFORE_LOGS") <(echo "$AFTER_LOGS") | grep -v '^$' || true)"
NEW_DRY="$(printf '%s\n' "$NEW_LOGS" | grep -c '_dryrun\.log$' || true)"
NEW_TRAIN="$(printf '%s\n' "$NEW_LOGS" | grep -cE '_(train|smoke)\.log$' || true)"
echo "  new logs: $(printf '%s ' $NEW_LOGS)"
check "G6a the dry run created exactly one _dryrun.log" "$([ "$NEW_DRY" = "1" ] && echo 0 || echo 1)"
check "G6b the dry run created NO _train.log or _smoke.log" "$([ "$NEW_TRAIN" = "0" ] && echo 0 || echo 1)"

echo "--- H. the exercise created nothing under outputs_FLAC/ ---"
if [ "$SAVEDIR_EXISTED" = "1" ]; then
  echo "SKIP  ${SAVEDIR} already existed before this exercise - not asserting its absence"
else
  check "H1 ${SAVEDIR} was not created by any gate" "$([ -e "$SAVEDIR" ] && echo 1 || echo 0)"
fi
if [ "$SMOKE_SAVEDIR_EXISTED" = "1" ]; then
  echo "SKIP  ${SMOKE_SAVEDIR} already existed before this exercise"
else
  check "H2 ${SMOKE_SAVEDIR} was not created by the smoke dry run" "$([ -e "$SMOKE_SAVEDIR" ] && echo 1 || echo 0)"
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
