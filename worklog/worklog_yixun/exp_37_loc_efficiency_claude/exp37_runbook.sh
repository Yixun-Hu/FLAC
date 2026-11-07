#!/usr/bin/env bash
# exp_37 loc_efficiency -- the EXECUTABLE runbook (fix rounds 1-2; Codex code reviews 1 / 2). NOT run by the Coder.
#
#   bash exp37_runbook.sh list               the registered sequence: id, phase, kind (? = optional), exact command
#   bash exp37_runbook.sh all                every remaining step in order; stops at the first failure (its status)
#   bash exp37_runbook.sh run STEP [STEP..]  named steps, in order (each must be next in the registered sequence)
#   bash exp37_runbook.sh skip STEP          the Planner's K_ctx 1 split decision only (B1.split_k1.{P1,YAW,BF}); the
#                                            conditional steps after it (TR.split_k1.*, B1S.all / B1S.noK1) are run or
#                                            recorded as skipped by run / all from those decisions
#   bash exp37_runbook.sh clean STEP         quarantine a FAILED model step's remains (tools/exp37_quarantine.py)
#   bash exp37_runbook.sh check-leaf         the B3 leaf worktree is at its launch commit with a clean tree
#
# Every executed step: refuses (exit 2) a dirty tracked tree, untracked content outside the allowlist, a missing or
# foreign predecessor receipt (every earlier step done; every earlier CAMPAIGN step -- P0, the gates, the model and
# leaf steps -- at this HEAD; the CPU consumers G0 cmp / B1S / B3S.cells / AGG may follow at a later HEAD), an existing
# output or a missing input; logs HEAD + the exact command into a NEW log (a logging failure is exit 74, the command's
# own nonzero status wins); a timed / selector / leaf step first runs the exp_36 gate (a unique record under
# $D/gates/pre/), writes that gate's receipt bound to the record ($D/receipts/pregate/) and checks it fresh at this
# HEAD -- an unclean gate refuses before the command; returns the command's own exit status; on success writes the
# hashed receipt $D/receipts/<step>.json. DRY_RUN=1 checks and prints without gating, running or receipting.
# Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix rounds 1-2.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1                  # the launcher requires the first
unset PYTHONPATH
SCRIPT_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
REPO_DIR=${REPO_DIR:-/home/yixunhu/codespace/exp-37-loc-efficiency}
PY=${PY:-/home/yixunhu/miniconda3/envs/flac/bin/python}
RPY=${RPY:-/home/yixunhu/miniconda3/envs/flac/bin/python}          # runs the receipt / quarantine tools
TOOLS=${TOOLS:-$SCRIPT_REPO/tools}
W=${W:-/media/diskstation/yixunhu/FLAC/exp37_pred_waveforms}
LEAF_ROOT=${LEAF_ROOT:-/home/yixunhu/codespace/exp-37-b3-synthesis}
LEAF_COMMIT=d30137349027914af7f04462b05347f0f9e03c02
LEAF_CHAIN=worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/yaw_c4_spread_chain.sh
DRY_RUN=${DRY_RUN:-0}
PREGATE_MAX_AGE_HOURS=0.1                                          # the pre-step gate receipt: <= 6 minutes old
D=worklog/worklog_yixun/exp_37_loc_efficiency_claude
E9=worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
LB=NeurIPS_2026_Workshop_Latency_Reproducibility
L=overlays/no_flash/exp37_no_flash_launch.py
S128=$LB/data/selection/frozen_16room_128.json
P050=$E9/pilot_manifest_seed42_4_per_room.json
A050=$E9/geometry_audit.json
P075=$D/audit_0p75/pilot_manifest_0p75_seed42_4_per_room.json
A075=$D/audit_0p75/geometry_audit_0p75.json
N100=$D/manifests/nested_1m_pilot_seed42_4_per_room.json
UNTRACKED_OK=(AcousticRooms "$D/runs/" "$D/selector/" "$D/gates/" "$D/g0/" "$D/b1/" "$D/b2b3/" "$D/receipts/")
declare -A CKPT=([P1]=weights/exp20/P1_40k.ckpt [YAW]=weights/exp20/YAW_40k.ckpt [BF]=weights/exp20/BF_40k.ckpt)
declare -A COND=([P1]=vanilla [YAW]=vanilla [BF]=fa_invariant)
IDS=() ; declare -A PHASE RUNNER ARGS OPTIONAL

add() { local id=$1 phase=$2 runner=$3; shift 3; IDS+=("$id"); PHASE[$id]=$phase; RUNNER[$id]=$runner; ARGS[$id]="$*"
        OPTIONAL[$id]=0; }
optional() { OPTIONAL[$1]=$2; }                                   # decision (skip by hand) | conditional (follows)

# ================================================================ the registered sequence (98 steps)
add P0.1 P0 cpu bash $D/exp37_assets.sh verify "$REPO_DIR"
add P0.2 P0 cpu env CUDA_VISIBLE_DEVICES= $PY $L --preflight
add P0.3 P0 cpu env CUDA_VISIBLE_DEVICES= $PY -m pytest -q -p no:cacheprovider src/tests -k exp37
add P0.4 P0 gate p0
for arm in YAW P1 BF; do                                                       # G0: YAW first
  add G0.$arm.run G0 timed g0_$arm $arm $S128 $A050 42 --query-index 4467
  add G0.$arm.cmp G0 g0cmp $arm g0_$arm
done
for arm in YAW P1 BF; do add G0.$arm.replay G0 replay g0_${arm}_replay g0_$arm $arm $S128 $A050 42 --query-index 4467; done
add SEL.0 SEL gate sel
add SEL.1 SEL selector selector_128 $S128
add SEL.2 SEL selector selector_0p75 $P075
add SEL.3 SEL selector selector_nested_1m $N100
add B1.0 B1 gate b1
for arm in P1 YAW BF; do add B1.k8.$arm B1 timed b1_k8_${arm}_s42 $arm $S128 $A050 42; done
for arm in P1 YAW BF; do add B1.k1.$arm B1 timed b1_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1; done
for arm in P1 YAW BF; do add B1.split_k8.$arm B1 timed b1_split_k8_${arm}_s42 $arm $S128 $A050 42 --stage-split; done
for arm in P1 YAW BF; do
  add B1.split_k1.$arm B1 timed b1_split_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1 --stage-split
  optional B1.split_k1.$arm decision
done
add B2.0 B2 gate b2
for g in 0p50 0p75; do
  for arm in P1 BF; do
    if [ $g = 0p50 ]; then sel=$P050 audit=$A050; else sel=$P075 audit=$A075; fi
    add B2.$g.$arm.s42 B2 timed b2_${g}_${arm}_s42 $arm $sel $audit 42
  done
done
for arm in P1 BF; do
  add B2.nested.$arm B2 timed b2t_nested_${arm}_s42 $arm $P050 $A050 42 --candidate-subset nested_1m
  add B2.c2f.$arm B2 timed b2t_c2f_${arm}_s42 $arm $P050 $A050 42 --candidate-subset c2f --c2f-top-k 5
done
add B3.0 B3 gate b3
add B3.lat.o0 B3 timed b3_lat_o0_s42 BF $S128 $A050 42 --frame-avg-angles 0
add B3.lat.o0_180 B3 timed b3_lat_o0_180_s42 BF $S128 $A050 42 --frame-avg-angles 0 180
# ---- TR: the MANDATORY artifact replays of every timing pass (announcement 08; no written exp_37 exemption)
for arm in P1 YAW BF; do
  add TR.b1_k8.$arm TR replay b1_k8_${arm}_s42_replay b1_k8_${arm}_s42 $arm $S128 $A050 42
  add TR.b1_k1.$arm TR replay b1_k1_${arm}_s42_replay b1_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1
  add TR.split_k8.$arm TR replay b1_split_k8_${arm}_s42_replay b1_split_k8_${arm}_s42 $arm $S128 $A050 42
  add TR.split_k1.$arm TR replay b1_split_k1_${arm}_s42_replay b1_split_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1
  optional TR.split_k1.$arm conditional
done
for g in 0p50 0p75; do
  for arm in P1 BF; do
    if [ $g = 0p50 ]; then sel=$P050 audit=$A050; else sel=$P075 audit=$A075; fi
    add TR.b2.$g.$arm TR replay b2_${g}_${arm}_s42_replay b2_${g}_${arm}_s42 $arm $sel $audit 42
  done
done
for arm in P1 BF; do
  add TR.nested.$arm TR replay b2t_nested_${arm}_s42_replay b2t_nested_${arm}_s42 $arm $P050 $A050 42 --candidate-subset nested_1m
  add TR.c2f.$arm TR replay b2t_c2f_${arm}_s42_replay b2t_c2f_${arm}_s42 $arm $P050 $A050 42 --candidate-subset c2f --c2f-top-k 5
done
add TR.lat.o0 TR replay b3_lat_o0_s42_replay b3_lat_o0_s42 BF $S128 $A050 42 --frame-avg-angles 0
add TR.lat.o0_180 TR replay b3_lat_o0_180_s42_replay b3_lat_o0_180_s42 BF $S128 $A050 42 --frame-avg-angles 0 180
# ---- the accuracy-only runs (inline dumps): B2 seeds 43 / 44 (D-a), the c2f accuracy runs, B3
for seed in 43 44; do
  for g in 0p50 0p75; do
    for arm in P1 BF; do
      if [ $g = 0p50 ]; then sel=$P050 audit=$A050; else sel=$P075 audit=$A075; fi
      add B2A.$g.$arm.s$seed B2A acc b2_${g}_${arm}_s$seed $arm $sel $audit $seed
    done
  done
done
for seed in 43 44; do
  for arm in P1 BF; do
    add B2A.c2f.$arm.s$seed B2A acc b2a_c2f_${arm}_s$seed $arm $P050 $A050 $seed --candidate-subset c2f --c2f-top-k 5
  done
done
for seed in 42 43 44; do
  add B3A.o0_t0.s$seed B3A acc b3_o0_t0_s$seed BF $P050 $A050 $seed --frame-avg-angles 0
  add B3A.o0_180_t0.s$seed B3A acc b3_o0_180_t0_s$seed BF $P050 $A050 $seed --frame-avg-angles 0 180
  add B3A.o0_t90.s$seed B3A acc b3_o0_t90_s$seed BF $P050 $A050 $seed --frame-avg-angles 0 --rotate-deg 90
  add B3A.o0_180_t90.s$seed B3A acc b3_o0_180_t90_s$seed BF $P050 $A050 $seed --frame-avg-angles 0 180 --rotate-deg 90
  add B3A.c4_t90.s$seed B3A acc b3_c4_t90_s$seed BF $P050 $A050 $seed --rotate-deg 90
done
# ---- the B1 summary: every pass + its replay (exactly one of the two variants runs: with / without the K1 splits)
B1RUNS=() B1REPLAYS=()
for arm in P1 YAW BF; do for k in 8 1; do
  B1RUNS+=(--run $arm $k $D/runs/b1_k${k}_${arm}_s42); B1REPLAYS+=(--replay $arm $k $D/runs/b1_k${k}_${arm}_s42_replay)
done; done
B1K8=() B1K1=()
for arm in P1 YAW BF; do
  B1K8+=(--split $arm 8 $D/runs/b1_split_k8_${arm}_s42 --split-replay $arm 8 $D/runs/b1_split_k8_${arm}_s42_replay)
  B1K1+=(--split $arm 1 $D/runs/b1_split_k1_${arm}_s42 --split-replay $arm 1 $D/runs/b1_split_k1_${arm}_s42_replay)
done
B1X=(--selection $S128 --selector $D/selector/selector_128.json --geometry-audit $A050)
add B1S.all B1S cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_summarize_latency.py ${B1X[*]} ${B1RUNS[*]} ${B1REPLAYS[*]} \
  ${B1K8[*]} ${B1K1[*]} --require-replays --output-json $D/b1/b1_latency_summary.json --output-md $D/b1/b1_latency_summary.md
optional B1S.all conditional
add B1S.noK1 B1S cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_summarize_latency.py ${B1X[*]} ${B1RUNS[*]} ${B1REPLAYS[*]} \
  ${B1K8[*]} --require-replays --output-json $D/b1/b1_latency_summary_noK1split.json \
  --output-md $D/b1/b1_latency_summary_noK1split.md
optional B1S.noK1 conditional
# ---- B3 synthesis: the Planner's leaf (chain b3 at its launch commit), after every exp_37 timing pass
add B3S.gate B3S gate b3s
add B3S.run B3S leaf
add B3S.cells B3S cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_build_synthesis_cells.py --output $D/b2b3/synthesis_cells.json
add AGG.spec AGG cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_build_b2b3_spec.py --inputs $D/b2b3/b2b3_inputs.json \
  --runs-root $D/runs --synthesis $D/b2b3/synthesis_cells.json --require-timing-replays --output $D/b2b3/b2b3_spec.json
add AGG.run AGG cpu env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=$D/b2b3/.mpl $PY tools/exp37_aggregate_b2b3.py \
  --spec $D/b2b3/b2b3_spec.json --output-json $D/b2b3/b2b3_summary.json --output-md $D/b2b3/b2b3_summary.md \
  --figure-stem $D/b2b3/b2_pareto

# ================================================================ the engine
LOCALIZE=(--model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --agree-ckpt weights/AGREE/AGREE_fullAR.pt
          --context-manifest $E9/context_manifest_exp01_seed42.json --dataset-root AcousticRooms
          --mesh-root AcousticRooms/room_mesh_obj_format --device cuda:0 --candidate-batch-size 64 --tau 0.1
          --score-sample-counts 1 --require-attention-backend sdpa)
TIMING=(--synchronize-latency --warmup-query-count 1 --measure-core-forward)

refuse() { echo "REFUSED [$1]: $2" >&2; return 2; }
receipt() { echo "$D/receipts/$1.json"; }
has_receipt() { "$RPY" "$TOOLS/exp37_receipt.py" check --path "$(receipt "$1")" --step "$1" ${2:+--status "$2"} \
                ${3:+--head "$3"} ${4:+--max-age-hours "$4"} 2>/dev/null; }
kind_of() {
  local k=${RUNNER[$1]}
  case $k in acc|replay) k=untimed ;; g0cmp) k=cpu ;; esac
  [ "${OPTIONAL[$1]}" = 0 ] || k="$k?"
  echo "$k"
}
model_step() { case ${RUNNER[$1]} in timed|acc|replay|selector|leaf) return 0 ;; esac; return 1; }
campaign_step() { model_step "$1" || [ "${RUNNER[$1]}" = gate ] || [ "${PHASE[$1]}" = P0 ]; }
gated_step() { case ${RUNNER[$1]} in timed|selector|leaf) return 0 ;; esac; return 1; }

# build_cmd STEP -> CMD (argv), OUTS (must not exist), NEEDS (must exist)
build_cmd() {
  local id=$1 kind run arm sel audit seed of; read -r -a a <<< "${ARGS[$id]}"; kind=${RUNNER[$id]}
  CMD=() OUTS=() NEEDS=()
  case $kind in
    cpu) CMD=("${a[@]}") ;;
    gate) CMD=(env CUDA_VISIBLE_DEVICES= $PY tools/exp37_cotenancy_monitor.py --gate-only --output $D/gates/${a[0]}.json)
          OUTS=($D/gates/${a[0]}.json) ;;
    selector) CMD=(env CUDA_VISIBLE_DEVICES=0 $PY $L tools/exp37_selector_latency.py --selection ${a[1]}
                   --context-manifest $E9/context_manifest_exp01_seed42.json --dataset-root AcousticRooms
                   --agree-checkpoint weights/AGREE/AGREE_fullAR.pt --device cuda:0 --candidate-batch-size 64
                   --repeat-count 3 --output $D/selector/${a[0]}.json)
              OUTS=($D/selector/${a[0]}.json $D/selector/${a[0]}.json.cotenancy) ;;
    g0cmp) CMD=(env CUDA_VISIBLE_DEVICES= $PY tools/exp37_g0_compare.py --arm ${a[0]} --run-dir $D/runs/${a[1]}
                --output $D/g0/g0_${a[0]}_verdict.json)
           [ "${a[0]}" = YAW ] || CMD+=(--yaw-verdict $D/g0/g0_YAW_verdict.json)
           OUTS=($D/g0/g0_${a[0]}_verdict.json); NEEDS=($D/runs/${a[1]}/run_manifest.json) ;;
    timed|acc) run=${a[0]} arm=${a[1]} sel=${a[2]} audit=${a[3]} seed=${a[4]}
               CMD=(env CUDA_VISIBLE_DEVICES=0 $PY $L localize_FLAC.py --ckpt-path ${CKPT[$arm]} --cond-method ${COND[$arm]}
                    --pilot-manifest $sel --geometry-audit $audit --sample-seed $seed --output-dir $D/runs/$run
                    "${LOCALIZE[@]}" "${a[@]:5}")
               OUTS=($D/runs/$run)
               if [ $kind = timed ]; then CMD+=("${TIMING[@]}"); else CMD+=(--dump-waveforms $W/$run); OUTS+=($W/$run); fi ;;
    replay) run=${a[0]} of=${a[1]} arm=${a[2]} sel=${a[3]} audit=${a[4]} seed=${a[5]}
            CMD=(env CUDA_VISIBLE_DEVICES=0 $PY $L localize_FLAC.py --ckpt-path ${CKPT[$arm]} --cond-method ${COND[$arm]}
                 --pilot-manifest $sel --geometry-audit $audit --sample-seed $seed --output-dir $D/runs/$run
                 "${LOCALIZE[@]}" "${a[@]:6}" --dump-waveforms $W/$run --replay-of $D/runs/$of)
            OUTS=($D/runs/$run $W/$run); NEEDS=($D/runs/$of/run_manifest.json) ;;
    leaf) CMD=(cd "$LEAF_ROOT" "&&" env "EXPECT_HEAD=$LEAF_COMMIT" bash $LEAF_CHAIN b3) ;;
    *) refuse "$id" "unknown step kind $kind"; return 2 ;;
  esac
}

check_leaf() {
  [ "$(git -C "$LEAF_ROOT" rev-parse HEAD 2>/dev/null)" = "$LEAF_COMMIT" ] || {
    refuse leaf "$LEAF_ROOT is not at the leaf launch commit $LEAF_COMMIT"; return 2; }
  [ -z "$(git --no-optional-locks -C "$LEAF_ROOT" status --porcelain --untracked-files=no)" ] || {
    refuse leaf "the leaf worktree's tracked tree is dirty"; return 2; }
}

# the condition of a conditional step: 0 holds, 1 does not (run / all record it as skipped), 3 none registered;
# exactly one B1S variant holds
condition() {
  local id=$1 arm
  case $id in
    TR.split_k1.*) has_receipt "B1.split_k1.${id##*.}" ok && return 0; return 1 ;;
    B1S.all) for arm in P1 YAW BF; do has_receipt B1.split_k1.$arm ok || return 1; done; return 0 ;;
    B1S.noK1) for arm in P1 YAW BF; do has_receipt B1.split_k1.$arm skipped && return 0; done; return 1 ;;
    *) refuse "$id" "no condition is registered"; return 3 ;;
  esac
}

preflight() {   # the checks every executed or skipped step passes first
  local id=$1 prior line path ok head
  [ -n "${PHASE[$id]:-}" ] || { refuse "$id" "no such step (bash $0 list)"; return 2; }
  cd "$REPO_DIR" || { refuse "$id" "no checkout at $REPO_DIR"; return 2; }
  ! has_receipt "$id" || { refuse "$id" "already done (its receipt exists)"; return 2; }
  [ -z "$(git --no-optional-locks status --porcelain --untracked-files=no)" ] || {
    refuse "$id" "the tracked tree is dirty"; return 2; }
  while IFS= read -r line; do
    path=${line#?? }; ok=0
    for prefix in "${UNTRACKED_OK[@]}"; do case $path in "$prefix"|"$prefix"*) ok=1 ;; esac; done
    case $path in $D/loc_efficiency_*.log) ok=1 ;; esac
    [ $ok = 1 ] || { refuse "$id" "untracked content outside the allowlist: $path"; return 2; }
  done < <(git --no-optional-locks status --porcelain --untracked-files=normal | grep '^?? ')
  head=$(git rev-parse HEAD)
  for prior in "${IDS[@]}"; do
    [ "$prior" = "$id" ] && break
    has_receipt "$prior" || { refuse "$id" "predecessor $prior has no receipt (the registered order)"; return 2; }
    if campaign_step "$prior" && campaign_step "$id"; then
      has_receipt "$prior" any "$head" || { refuse "$id" "the campaign step $prior ran at another HEAD"; return 2; }
    fi
  done
  HEAD_NOW=$head
}

# pregate STEP LOG: the exp_36 gate immediately before a timed / selector / leaf step, and its fresh bound receipt
pregate() {
  local id=$1 log=$2 stamp record receipt grc started
  stamp=$(date -u +%Y%m%dT%H%M%S%NZ); record=$D/gates/pre/$id.$stamp.json; receipt=$D/receipts/pregate/$id.$stamp.json
  started=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
  env CUDA_VISIBLE_DEVICES= $PY tools/exp37_cotenancy_monitor.py --gate-only --output "$record" 2>&1 | tee -a "$log"
  local status=("${PIPESTATUS[@]}"); grc=${status[0]}; local trc=${status[1]}
  echo "PREGATE EXIT $grc $record" | tee -a "$log" >/dev/null || trc=74
  [ "$grc" = 0 ] || { refuse "$id" "the pre-step exp_36 gate is not clean (exit $grc; $record)"; return 2; }
  [ "$trc" = 0 ] || { echo "REFUSED [$id]: the audit log failed (pre-step gate)" >&2; return 74; }
  "$RPY" "$TOOLS/exp37_receipt.py" write --path "$receipt" --step "$id.pregate" --phase "${PHASE[$id]}" --status ok \
    --head "$HEAD_NOW" --command "env CUDA_VISIBLE_DEVICES= $PY tools/exp37_cotenancy_monitor.py --gate-only" \
    --log "$log" --started "$started" --finished "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" --record "$record" || {
    refuse "$id" "the pre-step gate receipt was not written"; return 2; }
  "$RPY" "$TOOLS/exp37_receipt.py" check --path "$receipt" --step "$id.pregate" --status ok --head "$HEAD_NOW" \
    --max-age-hours "$PREGATE_MAX_AGE_HOURS" || { refuse "$id" "no fresh pre-step gate receipt"; return 2; }
}

record_skip() {
  local id=$1 why=$2 now; now=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
  "$RPY" "$TOOLS/exp37_receipt.py" write --path "$(receipt "$id")" --step "$id" --phase "${PHASE[$id]}" --status skipped \
    --head "$HEAD_NOW" --command "skipped: $why" --log none --started "$now" --finished "$now" || return 2
  echo "SKIPPED [$id]: $why"
}

run_step() {
  local id=$1 log rc trc started path
  preflight "$id" || return $?
  if [ "${OPTIONAL[$id]}" = conditional ]; then
    local holds=0; condition "$id" || holds=$?
    [ "$holds" -le 1 ] || return 2                                  # no registered condition: fail closed
    if [ "$holds" = 1 ]; then
      [ "$DRY_RUN" = 1 ] && { echo "DRY_RUN: $id would be recorded as skipped"; return 0; }
      record_skip "$id" "its condition does not hold (the K_ctx 1 split decisions)"; return $?
    fi
  fi
  build_cmd "$id" || return 2
  [ "${RUNNER[$id]}" != leaf ] || check_leaf || return 2
  for path in "${OUTS[@]}"; do [ ! -e "$path" ] || { refuse "$id" "$path exists (outputs are fresh)"; return 2; }; done
  for path in "${NEEDS[@]}"; do [ -e "$path" ] || { refuse "$id" "$path is missing"; return 2; }; done
  mkdir -p "$D" 2>/dev/null
  local base n=0; base=$D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_${id}; log=$base.log
  until ( set -C; : > "$log" ) 2>/dev/null; do                      # a NEW log per attempt (exclusive create)
    n=$((n + 1)); [ "$n" -le 20 ] || { echo "REFUSED [$id]: cannot create the audit log $log" >&2; return 74; }
    log=${base}_$n.log
  done
  started=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
  { echo "HEAD $HEAD_NOW step=$id phase=${PHASE[$id]}"; printf 'COMMAND'; printf ' %q' "${CMD[@]}"; echo; } | tee -a "$log"
  [ "${PIPESTATUS[1]}" = 0 ] || { echo "REFUSED [$id]: the audit log failed" >&2; return 74; }
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: not executed" | tee -a "$log"; return 0; fi
  if gated_step "$id"; then pregate "$id" "$log" || return $?; fi
  if [ "${RUNNER[$id]}" = leaf ]; then
    ( cd "$LEAF_ROOT" && env "EXPECT_HEAD=$LEAF_COMMIT" bash $LEAF_CHAIN b3 ) 2>&1 | tee -a "$log"
  else
    "${CMD[@]}" 2>&1 | tee -a "$log"
  fi
  local status=("${PIPESTATUS[@]}"); rc=${status[0]}; trc=${status[1]}
  echo "EXIT $rc" | tee -a "$log" >/dev/null || trc=74
  [ "$rc" = 0 ] || return "$rc"
  [ "$trc" = 0 ] || { echo "REFUSED [$id]: the audit log failed (the command exited 0)" >&2; return 74; }
  "$RPY" "$TOOLS/exp37_receipt.py" write --path "$(receipt "$id")" --step "$id" --phase "${PHASE[$id]}" --status ok \
    --head "$HEAD_NOW" --command "$(printf '%q ' "${CMD[@]}")" --log "$log" --started "$started" \
    --finished "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" || { echo "REFUSED [$id]: the receipt was not written" >&2; return 74; }
}

skip_step() {
  local id=$1
  [ "${OPTIONAL[$id]:-0}" = decision ] || {
    refuse "$id" "only the K_ctx 1 split decisions B1.split_k1.{P1,YAW,BF} are skipped by hand"; return 2; }
  preflight "$id" || return $?
  record_skip "$id" "the Planner's K_ctx 1 split decision"
}

case ${1:-} in
  list) for id in "${IDS[@]}"; do build_cmd "$id"; printf '%-22s %-5s %-9s' "$id" "${PHASE[$id]}" "$(kind_of "$id")"
          printf ' %q' "${CMD[@]}"; echo; done ;;
  run) shift; [ "$#" -gt 0 ] || { echo "usage: bash $0 run STEP [STEP ...]" >&2; exit 2; }
       for id in "$@"; do run_step "$id"; rc=$?; [ "$rc" = 0 ] || { echo "STOPPED at $id (exit $rc)" >&2; exit "$rc"; }; done ;;
  all) for id in "${IDS[@]}"; do
         (cd "$REPO_DIR" && has_receipt "$id") && continue
         run_step "$id"; rc=$?
         [ "$rc" = 0 ] || { echo "STOPPED at $id (exit $rc)" >&2; exit "$rc"; }
       done ;;
  skip) [ -n "${2:-}" ] || { echo "usage: bash $0 skip STEP" >&2; exit 2; }; skip_step "$2"; exit $? ;;
  clean) id=${2:-}
         case ${RUNNER[$id]:-} in timed|acc|replay) ;; *) refuse "$id" "only a model step can be cleaned"; exit 2 ;; esac
         cd "$REPO_DIR" || exit 2
         ! has_receipt "$id" || { refuse "$id" "a completed step is never cleaned"; exit 2; }
         build_cmd "$id"
         "$RPY" "$TOOLS/exp37_quarantine.py" --run-dir "${OUTS[0]}" --dump-root "$W" ${3:+"$3"}; exit $? ;;
  check-leaf) check_leaf; exit $? ;;
  *) echo "usage: bash $0 list | all | run STEP [STEP ...] | skip STEP | clean STEP [--include-published] | check-leaf" >&2
     exit 2 ;;
esac
