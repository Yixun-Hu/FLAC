#!/usr/bin/env bash
# ============================================================================
# haa_ft_eval.sh — exp_19: the HAA finetune evaluation grid.
#
#   EXPECT_SHA=<HEAD> bash haa_ft_eval.sh            # all three arms
#   ARMS="P1" EXPECT_SHA=<HEAD> bash haa_ft_eval.sh  # one arm as it finishes
#   DRY_RUN=1 bash haa_ft_eval.sh                    # print the queue, run nothing
#
# The grid is 3 arms x 2 endpoints x 2 K x 5 seeds = 60 cells:
#
#   arms      P1, BF, YAW
#   endpoints step-410 and step-1000 (plan B1 keeps BOTH readings)
#   K         8 -> haa_test.json, 1 -> haa_test_1.json   (K is a property of the
#             DATASET CONFIG, never a flag — pairing a K label with the wrong
#             config mislabels every number in the row)
#   seeds     42..46, the same five every 40k row in the record was produced with
#
# **Per-arm eval protocol is part of the experiment** (announcement 05). The flag
# must match how the checkpoint was TRAINED; a mismatch produces plausible,
# catastrophically wrong numbers in both directions and has already cost this
# repo one retracted conclusion:
#
#   P1, YAW   --cond-method vanilla          (and NO orbit flags at all)
#   BF        --cond-method fa_invariant --frame-avg-angles 0,90,180,270
#             --frame-avg-max-fwd-samples 64 (announcement 06: the chunk plan is
#             declared, not inherited; 64 is what every row in the record used)
#
# All 60 cells: --cond-autocast bf16 --cfg-scale 1.0 --steps 1, and
# --record-per-scene so the PER-SCENE mean — this repo's headline convention and
# the plan's estimand (§2.4) — is computable from the artifact. The flat metrics
# block is unaffected by that flag.
#
# eval_FLAC.py writes its metric JSON next to the checkpoint it reads. Here that
# is our OWN namespace (outputs_FLAC/exp19_HAA_<ARM>/...), so no symlink farm is
# needed — unlike exp_17, which evaluated a foreign worktree's checkpoints.
#
# Resume is per cell and is decided by PARSING the existing record, never by its
# filename: a JSON whose cond_method / cond_autocast / seed / split / orbit cap
# do not match what this cell would produce is re-run, because a filename cannot
# testify to the protocol that produced it.
#
# Written by the exp_19 coder seat (Claude Opus 5, max effort).
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/../../.." || { echo "cannot reach repo root - abort"; exit 2; }
EXPDIR="worklog/worklog_yixun/exp_19_haa_finetune_claude"

# --- inputs ------------------------------------------------------------------ #
ARMS="${ARMS:-P1 BF YAW}"
DRY_RUN="${DRY_RUN:-0}"
EXPECT_SHA="${EXPECT_SHA:-}"
MIN_FREE_MB="${MIN_FREE_MB:-}"
DEFAULT_MIN_FREE_MB=8000          # one eval process per card, bf16 conditioning

# Test-only overrides: each substitutes EVIDENCE for a gate, so each is refused
# in production (DRY_RUN=0) and disclosed loudly when used.
CKPT_ROOT_DEFAULT="outputs_FLAC"
CKPT_ROOT="${CKPT_ROOT:-$CKPT_ROOT_DEFAULT}"
EVAL_CMD="${EVAL_CMD:-}"                      # replaces `python eval_FLAC.py`
TRAIN_PGREP_DEFAULT='[t]rain\.py'             # bracket trick: cannot match pgrep itself
TRAIN_PGREP="${TRAIN_PGREP:-$TRAIN_PGREP_DEFAULT}"
GATE_LOCK_HELD_BY_CALLER="${GATE_LOCK_HELD_BY_CALLER:-}"

# --- pinned constants -------------------------------------------------------- #
STEPS_GRID=(410 1000)
SEEDS=(42 43 44 45 46)
K8_CFG="src/configs/dataset_configs/HAA/eval/haa_test.json"
K1_CFG="src/configs/dataset_configs/HAA/eval/haa_test_1.json"
STOCK_CFG="src/configs/model_configs/FLAC/HAA/FLAC_HAA_finetune.json"
CFG_SCALE="1.0"
DIFF_STEPS=1
AUTOCAST="bf16"
FA_ANGLES="0,90,180,270"
FA_CAP=64
EXPECTED_CELLS=60

case "$DRY_RUN" in
  0|1) ;;
  *) echo "DRY_RUN must be exactly 0 or 1 (got '${DRY_RUN}') - abort"; exit 2 ;;
esac
for A in $ARMS; do
  case "$A" in
    P1|BF|YAW|YNA) ;;
    *) echo "ARMS must be a subset of 'P1 BF YAW' (got token '${A}') - abort"; exit 2 ;;
  esac
done

# --- gate 1: the test overrides are refused in PRODUCTION --------------------- #
# Checked before EXPECT_SHA so each guard case asserts its own message, with the
# revision gate as the independent second refusal behind it.
if [ "$DRY_RUN" = "0" ]; then
  [ -z "$EVAL_CMD" ] || { echo "EVAL_CMD is a DRY_RUN-only test override and is set ('${EVAL_CMD}') - a real queue must run eval_FLAC.py - abort"; exit 2; }
  [ "$CKPT_ROOT" = "$CKPT_ROOT_DEFAULT" ] || { echo "CKPT_ROOT is a DRY_RUN-only test override and points at '${CKPT_ROOT}' instead of ${CKPT_ROOT_DEFAULT} - abort"; exit 2; }
  [ "$TRAIN_PGREP" = "$TRAIN_PGREP_DEFAULT" ] || { echo "TRAIN_PGREP is a DRY_RUN-only test override and is set ('${TRAIN_PGREP}') - a real queue must look for the real train.py - abort"; exit 2; }
  [ -z "$GATE_LOCK_HELD_BY_CALLER" ] || { echo "GATE_LOCK_HELD_BY_CALLER is a DRY_RUN-only test override and is set - a real queue must acquire the lock itself - abort"; exit 2; }
fi

# --- gate 2: revision binding ------------------------------------------------- #
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
if [ "$DRY_RUN" = "0" ]; then
  [ -n "$EXPECT_SHA" ] || {
    echo "EXPECT_SHA is REQUIRED for a production eval queue: name the reviewed revision (git rev-parse HEAD) so these 60 rows are bound to it - abort"; exit 2; }
  [ -n "$HEAD_SHA" ] || { echo "git rev-parse HEAD failed - refusing to evaluate from an unidentifiable revision - abort"; exit 2; }
  [ "$EXPECT_SHA" = "$HEAD_SHA" ] || {
    echo "EXPECT_SHA=${EXPECT_SHA} != HEAD ${HEAD_SHA} - this checkout is not the revision you reviewed - abort"; exit 2; }
fi
MIN_FREE_MB="${MIN_FREE_MB:-$DEFAULT_MIN_FREE_MB}"
if [ "$MIN_FREE_MB" -lt "$DEFAULT_MIN_FREE_MB" ] 2>/dev/null; then
  echo "MIN_FREE_MB=${MIN_FREE_MB} is below the registered floor ${DEFAULT_MIN_FREE_MB}; CLAMPED (floors may be raised, never lowered)"
  MIN_FREE_MB="$DEFAULT_MIN_FREE_MB"
fi

TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/haa_ft_eval_${TS}.log"
[ "$DRY_RUN" = "1" ] && LOG="$(mktemp -t haa_ft_eval_dryrun.XXXXXXXX.log)"

# --- gate 3: one queue at a time, for the WHOLE queue ------------------------- #
# Not gate-phase-scoped like the launcher's: two queues would schedule the same
# cells twice, put two evaluators on one card, and race writes to one JSON.
LOCK="${EXPDIR}/.haa_eval.lock"
if [ -n "$GATE_LOCK_HELD_BY_CALLER" ]; then
  LOCK_OWNED=0
else
  exec 9>"$LOCK" || { echo "cannot open ${LOCK} - abort"; exit 2; }
  flock -n 9 || { echo "another exp_19 eval queue holds ${LOCK} - abort"; exit 2; }
  LOCK_OWNED=1
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== exp_19 HAA eval grid | arms='${ARMS}' | ${TS} | HEAD ${HEAD_SHA:-unknown} | dry_run=${DRY_RUN} ==="
echo "protocol: cfg=${CFG_SCALE} steps=${DIFF_STEPS} autocast=${AUTOCAST} seeds=${SEEDS[*]} endpoints=${STEPS_GRID[*]} | per-scene recorded"
echo "lock: ${LOCK} owned=${LOCK_OWNED}"
if [ "$DRY_RUN" = "1" ]; then
  [ -n "$EVAL_CMD" ] && echo "!! TEST OVERRIDE ACTIVE: EVAL_CMD='${EVAL_CMD}'"
  [ "$CKPT_ROOT" != "$CKPT_ROOT_DEFAULT" ] && echo "!! TEST OVERRIDE ACTIVE: CKPT_ROOT='${CKPT_ROOT}'"
  [ "$TRAIN_PGREP" != "$TRAIN_PGREP_DEFAULT" ] && echo "!! TEST OVERRIDE ACTIVE: TRAIN_PGREP='${TRAIN_PGREP}'"
  [ -n "$GATE_LOCK_HELD_BY_CALLER" ] && echo "!! TEST OVERRIDE ACTIVE: the caller holds ${LOCK}"
  true
fi

# --- gate 4: nothing of OURS may still be training ---------------------------- #
# Scope is this worktree, decided by /proc/<pid>/cwd — CLAUDE.md is explicit that
# a train.py on this box may belong to a sibling checkout, and a blanket refusal
# would block on runs we neither own nor may touch. Ours is a hard refusal (its
# step-1000 checkpoint may not exist yet, or may be mid-write); foreign ones are
# DISCLOSED as co-tenants, because they contend for the same cards.
OURS="$(pwd -P)"; MINE=""; FOREIGN=""
for PID in $(pgrep -f "$TRAIN_PGREP" 2>/dev/null); do
  CWD="$(readlink -f "/proc/${PID}/cwd" 2>/dev/null)"
  if [ "$CWD" = "$OURS" ]; then MINE="${MINE} ${PID}"; else FOREIGN="${FOREIGN} ${PID}(${CWD:-?})"; fi
done
[ -z "$MINE" ] || {
  echo "REFUSING: train.py from THIS worktree is still running (pid(s)${MINE}). An arm that is"
  echo "  still training has no final step-1000 checkpoint, and evaluating beside it contends"
  echo "  for the same cards. Wait for it to finish."; exit 2; }
echo "gate: no train.py from this worktree is running"
[ -z "$FOREIGN" ] && echo "co-tenancy: no foreign train.py on this box" \
                  || echo "co-tenancy DISCLOSED: foreign train.py (other checkouts):${FOREIGN}"

# --- gate 5: source pins ------------------------------------------------------ #
# The evaluator, both K configs, the split inventory they delegate to, and every
# arm's model config. The arm-config pins are the launcher's, so a row can only
# be produced by the same bytes the arm was trained under.
PIN_eval="286d5f69188b72641cb104823cecc919f7c6a6eba79811b29c1d6fb616d55611  eval_FLAC.py"
PIN_k8="fe9c473784340d32e5e02cbfa62948c52a1a0d18bf9869606fdeb898b0de11fc  ${K8_CFG}"
PIN_k1="140e0ce8422e03c5f6f37c1681d0bc68c1ca6fa5821289f5d1f726a17a66a46b  ${K1_CFG}"
PIN_testsplit="d4e3e29fddeeab6d9343a7efa811f5c02e23c3e99bd44894898a45d2a33248a1  data/HAA/test_base.json"
PIN_p1cfg="3639a9face84d13bcbb8f4472e78970c8e045952337f11b4f77d8798f786ba80  ${STOCK_CFG}"
PIN_bfcfg="834e4933f2f5c8050f196043e11260e00023a7c31205a55961e0a77ca910c1dc  ${EXPDIR}/FLAC_HAA_finetune_BF.json"
PIN_yawcfg="a03d106cd72744df40187b5c493010ecc996275b2afa32a4811d7c962c77cb53  ${EXPDIR}/FLAC_HAA_finetune_YAW.json"
NPINS=0
for P in "$PIN_eval" "$PIN_k8" "$PIN_k1" "$PIN_testsplit" "$PIN_p1cfg" "$PIN_bfcfg" "$PIN_yawcfg"; do
  echo "$P" | sha256sum -c --status - || {
    echo "SOURCE PIN FAILED for '${P##*  }' - the evaluator, a split or an arm config moved - abort"; exit 2; }
  NPINS=$((NPINS+1))
done
echo "source pins OK (${NPINS} files: evaluator, both K configs, the test split, all three arm configs)"

# --- per-arm protocol --------------------------------------------------------- #
arm_config() {
  case "$1" in
    P1)  printf '%s' "$STOCK_CFG" ;;
    YNA) printf '%s' "$STOCK_CFG" ;;
    BF)  printf '%s' "${EXPDIR}/FLAC_HAA_finetune_BF.json" ;;
    YAW) printf '%s' "${EXPDIR}/FLAC_HAA_finetune_YAW.json" ;;
  esac
}
# The eval-name suffix eval_FLAC appends for a non-vanilla method: it derives
# `_<method>_a<n_angles>` itself, so the resume path must predict the SAME name
# rather than inventing one.
arm_suffix() { [ "$1" = "BF" ] && printf '%s' "_fa_invariant_a4" || printf '%s' ""; }

# --- gate 6: exactly one checkpoint per (arm, endpoint) ----------------------- #
# PL names them epoch=<N>-step=<M>.ckpt and the epoch prefix varies (the HAA
# split is tiny, so an epoch is a handful of steps). Zero matches means the arm
# has not reached that endpoint; two means the namespace holds more than one run
# and picking either would be a coin flip.
declare -A CKPT
for ARM in $ARMS; do
  for STEP in "${STEPS_GRID[@]}"; do
    mapfile -t FOUND < <(find "${CKPT_ROOT}/exp19_HAA_${ARM}" -path '*/checkpoints/*' \
                              -name "*step=${STEP}.ckpt" -type f 2>/dev/null | sort)
    [ "${#FOUND[@]}" -eq 1 ] || {
      echo "CHECKPOINT GLOB FAILED for ${ARM} step=${STEP}: ${#FOUND[@]} match(es) under ${CKPT_ROOT}/exp19_HAA_${ARM} (need exactly 1)"
      [ "${#FOUND[@]}" -gt 1 ] && printf '    %s\n' "${FOUND[@]}"
      echo "  0 means the arm has not reached that endpoint; >1 means the namespace holds more than one run - abort"; exit 2; }
    CKPT["${ARM}_${STEP}"]="${FOUND[0]}"
    echo "ckpt ${ARM} step=${STEP}: ${FOUND[0]}"
  done
done

# --- build the queue ---------------------------------------------------------- #
# A cell is SKIPPED only when an existing record proves it was produced under
# this exact protocol. Parsed, never inferred from the filename.
record_matches() {   # <json> <cond_method> <seed> <dataset_cfg>
  python - "$1" "$2" "$3" "$4" "$AUTOCAST" "$FA_CAP" "$FA_ANGLES" <<'PY' >/dev/null 2>&1
import json, sys
path, method, seed, dscfg, autocast, cap, angles = sys.argv[1:8]
r = json.load(open(path))
if not isinstance(r.get("metrics"), dict) or not r["metrics"]:
    sys.exit(1)                                   # an empty/partial record is not a result
checks = [r.get("cond_method") == method,
          r.get("cond_autocast") == autocast,
          str(r.get("seed")) == seed,
          r.get("dataset_config") == dscfg,
          r.get("rotate_deg") in (0, 0.0)]
if method == "fa_invariant":
    want = [float(a) for a in angles.split(",")]
    checks += [r.get("frame_avg_angles") == want, r.get("frame_avg_fwd_cap") == int(cap)]
else:
    checks += [r.get("frame_avg_angles") is None]
sys.exit(0 if all(checks) else 1)
PY
}

QUEUE=(); SKIPPED=0; PLANNED=0
for ARM in $ARMS; do
  ACFG="$(arm_config "$ARM")"; SUFFIX="$(arm_suffix "$ARM")"
  for STEP in "${STEPS_GRID[@]}"; do
    CK="${CKPT["${ARM}_${STEP}"]}"
    STEM="$(basename "$CK" .ckpt)"
    for K in 8 1; do
      [ "$K" = "8" ] && DCFG="$K8_CFG" || DCFG="$K1_CFG"
      for SEED in "${SEEDS[@]}"; do
        PLANNED=$((PLANNED+1))
        NAME="exp19_HAA_${ARM}_S${STEP}_K${K}_s${SEED}"
        JSON="$(dirname "$CK")/${STEM}_metrics_${DIFF_STEPS}_${CFG_SCALE}_${NAME}${SUFFIX}.json"
        if [ -s "$JSON" ] && record_matches "$JSON" \
             "$([ "$ARM" = "BF" ] && echo fa_invariant || echo vanilla)" "$SEED" "$DCFG"; then
          echo "SKIP: ${NAME} (existing record matches this protocol)"
          SKIPPED=$((SKIPPED+1)); continue
        fi
        [ -s "$JSON" ] && echo "RERUN: ${NAME} (a record exists but does NOT match this protocol)"
        ARGVLINE="--model-config ${ACFG} --dataset-config ${DCFG} --ckpt-path ${CK}"
        ARGVLINE="${ARGVLINE} --cfg-scale ${CFG_SCALE} --steps ${DIFF_STEPS} --seed ${SEED}"
        ARGVLINE="${ARGVLINE} --eval-name ${NAME} --cond-autocast ${AUTOCAST} --record-per-scene"
        if [ "$ARM" = "BF" ]; then
          # announcement 05: the arm was TRAINED frame-averaged, so it is
          # evaluated frame-averaged; announcement 06: the chunk plan is declared.
          ARGVLINE="${ARGVLINE} --cond-method fa_invariant --frame-avg-angles ${FA_ANGLES} --frame-avg-max-fwd-samples ${FA_CAP}"
        else
          # A vanilla cell is given NO orbit flags at all: a vanilla row that
          # carried them would look protocol-compatible with a frame-averaged one.
          ARGVLINE="${ARGVLINE} --cond-method vanilla"
        fi
        QUEUE+=("${NAME}"$'\t'"${ARGVLINE}")
      done
    done
  done
done

echo "queue: ${PLANNED} cells planned for arms '${ARMS}' (${SKIPPED} already recorded, ${#QUEUE[@]} to run)"
[ "$ARMS" != "P1 BF YAW" ] || [ "$PLANNED" -eq "$EXPECTED_CELLS" ] || {
  echo "GRID SIZE WRONG: planned ${PLANNED}, registered ${EXPECTED_CELLS} - abort"; exit 2; }

for ENTRY in "${QUEUE[@]}"; do
  echo "CELL: python eval_FLAC.py ${ENTRY#*$'\t'}"
done

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN: all gates passed; ${#QUEUE[@]} cell(s) NOT run"; exit 0
fi

# --- resource floor on both cards --------------------------------------------- #
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} - refusing to evaluate blind - abort"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - abort"; exit 2; }
done
echo "both cards clear the ${MIN_FREE_MB} MiB floor"

# --- run: two cells in flight, one per card ----------------------------------- #
# A bare `wait` returns 0 regardless of child failures (GNU bash manual), so each
# PID is waited on individually and failures are counted (exp_17 rev-1 defect).
i=0; FAILED=0; PIDS=(); LABELS=()
drain() {
  local j
  for j in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$j]}"; then
      echo "  !! cell FAILED: ${LABELS[$j]} (see ${EXPDIR}/eval_${LABELS[$j]}.log)"
      FAILED=$((FAILED+1))
    fi
  done
  PIDS=(); LABELS=()
}
for ENTRY in "${QUEUE[@]}"; do
  NAME="${ENTRY%%$'\t'*}"; CELLARGS="${ENTRY#*$'\t'}"
  GPU=$((i % 2))
  CELLLOG="${EXPDIR}/eval_${NAME}.log"
  echo "[$((i+1))/${#QUEUE[@]}] gpu${GPU} ${NAME}"
  # shellcheck disable=SC2086
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" python eval_FLAC.py $CELLARGS > "$CELLLOG" 2>&1 &
  PIDS+=("$!"); LABELS+=("$NAME")
  i=$((i+1))
  [ $((i % 2)) -eq 0 ] && drain
done
drain
echo "evaluator failures: ${FAILED}"

# --- completeness: recount from the ARTIFACTS, not from the loop -------------- #
DONE=0; MISSING=()
for ARM in $ARMS; do
  SUFFIX="$(arm_suffix "$ARM")"
  METHOD="$([ "$ARM" = "BF" ] && echo fa_invariant || echo vanilla)"
  for STEP in "${STEPS_GRID[@]}"; do
    CK="${CKPT["${ARM}_${STEP}"]}"; STEM="$(basename "$CK" .ckpt)"
    for K in 8 1; do
      [ "$K" = "8" ] && DCFG="$K8_CFG" || DCFG="$K1_CFG"
      for SEED in "${SEEDS[@]}"; do
        NAME="exp19_HAA_${ARM}_S${STEP}_K${K}_s${SEED}"
        JSON="$(dirname "$CK")/${STEM}_metrics_${DIFF_STEPS}_${CFG_SCALE}_${NAME}${SUFFIX}.json"
        if [ -s "$JSON" ] && record_matches "$JSON" "$METHOD" "$SEED" "$DCFG"; then
          DONE=$((DONE+1))
        else
          MISSING+=("$NAME")
        fi
      done
    done
  done
done
echo "=== exp_19 HAA eval: ${DONE}/${PLANNED} cells complete (registered grid ${EXPECTED_CELLS}) ==="
if [ "${#MISSING[@]}" -ne 0 ]; then
  echo "INCOMPLETE — missing or protocol-mismatched cells:"
  printf '    %s\n' "${MISSING[@]}"
  echo "re-run this script to resume (completed cells are skipped)"
  exit 1
fi
echo "all planned cells recorded under their arms' checkpoint directories"
exit 0
