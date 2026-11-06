#!/usr/bin/env bash
# exp_37 loc_efficiency -- the EXECUTABLE runbook (fix round 1; Codex code review 1, BLOCKING 5). NOT run by the Coder.
#
#   bash exp37_runbook.sh list              every step in execution order with its exact command (printf %q)
#   bash exp37_runbook.sh run STEP [STEP..] run the steps in order; stop at the first failure with ITS exit status
#   bash exp37_runbook.sh phase PHASE       run every step of one phase in order (same stop rule)
#
# Every step refuses (exit 2) on a dirty tracked tree or an existing output, records HEAD + the exact command in a
# timestamped log under the experiment folder, and returns the command's own exit status (PIPESTATUS, never masked).
# DRY_RUN=1 performs the checks and prints the command without running it. Model steps run through the no-flash
# launcher with --require-attention-backend sdpa on GPU 0; timed steps are monitored by the runner (cotenancy/),
# untimed steps dump their scored waveforms to $W/<run> (announcement 08), replays reproduce a timing pass bitwise.
# Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 1.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1                  # the launcher requires the first
unset PYTHONPATH
REPO_DIR=${REPO_DIR:-/home/yixunhu/codespace/exp-37-loc-efficiency}
PY=${PY:-/home/yixunhu/miniconda3/envs/flac/bin/python}
W=${W:-/media/diskstation/yixunhu/FLAC/exp37_pred_waveforms}
DRY_RUN=${DRY_RUN:-0}
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
SYNTHESIS_PENDING="PENDING -- external: the 8 new eval_FLAC synthesis cells ({0}, {0,180} x theta {0, 90} x K {1, 8}, \
seed 42, no-flash) run on the exp_36 Coder's H35-compatible leaf; the 4 C4 cells are reused from exp_35 phase (i). \
The Planner fills the leaf commit, its runbook commands and the reused paths here and in \
$D/b2b3/synthesis_cells.json (12 cells); until then AGG produces a DIAGNOSTIC only."
declare -A CKPT=([P1]=weights/exp20/P1_40k.ckpt [YAW]=weights/exp20/YAW_40k.ckpt [BF]=weights/exp20/BF_40k.ckpt)
declare -A COND=([P1]=vanilla [YAW]=vanilla [BF]=fa_invariant)
IDS=() ; declare -A PHASE ARGS

add() { local id=$1 phase=$2; shift 2; IDS+=("$id"); PHASE[$id]=$phase; ARGS[$id]="$*"; }

# ---------------------------------------------------------------- the steps, in execution order
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
for arm in P1 YAW BF; do                                                       # optional (the Planner's call)
  add B1.split_k1.$arm B1 timed b1_split_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1 --stage-split
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
# ---- untimed: the registered artifact replays of the timing passes whose accuracy is reported, then the accuracy runs
for g in 0p50 0p75; do
  for arm in P1 BF; do
    if [ $g = 0p50 ]; then sel=$P050 audit=$A050; else sel=$P075 audit=$A075; fi
    add B2R.$g.$arm B2R replay b2_${g}_${arm}_s42_replay b2_${g}_${arm}_s42 $arm $sel $audit 42
  done
done
for seed in 43 44; do
  for g in 0p50 0p75; do
    for arm in P1 BF; do
      if [ $g = 0p50 ]; then sel=$P050 audit=$A050; else sel=$P075 audit=$A075; fi
      add B2A.$g.$arm.s$seed B2A acc b2_${g}_${arm}_s$seed $arm $sel $audit $seed
    done
  done
done
for seed in 42 43 44; do
  add B3A.o0_t0.s$seed B3A acc b3_o0_t0_s$seed BF $P050 $A050 $seed --frame-avg-angles 0
  add B3A.o0_180_t0.s$seed B3A acc b3_o0_180_t0_s$seed BF $P050 $A050 $seed --frame-avg-angles 0 180
  add B3A.o0_t90.s$seed B3A acc b3_o0_t90_s$seed BF $P050 $A050 $seed --frame-avg-angles 0 --rotate-deg 90
  add B3A.o0_180_t90.s$seed B3A acc b3_o0_180_t90_s$seed BF $P050 $A050 $seed --frame-avg-angles 0 180 --rotate-deg 90
  add B3A.c4_t90.s$seed B3A acc b3_c4_t90_s$seed BF $P050 $A050 $seed --rotate-deg 90
done
# ---- CONDITIONAL: replays of the timing-only passes (needed only if Yixun does not exempt pure latency passes)
for arm in P1 YAW BF; do
  add XR.b1_k8.$arm XR replay b1_k8_${arm}_s42_replay b1_k8_${arm}_s42 $arm $S128 $A050 42
  add XR.b1_k1.$arm XR replay b1_k1_${arm}_s42_replay b1_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1
  add XR.split_k8.$arm XR replay b1_split_k8_${arm}_s42_replay b1_split_k8_${arm}_s42 $arm $S128 $A050 42
  add XR.split_k1.$arm XR replay b1_split_k1_${arm}_s42_replay b1_split_k1_${arm}_s42 $arm $S128 $A050 42 --context-count 1
done
for arm in P1 BF; do
  add XR.nested.$arm XR replay b2t_nested_${arm}_s42_replay b2t_nested_${arm}_s42 $arm $P050 $A050 42 --candidate-subset nested_1m
  add XR.c2f.$arm XR replay b2t_c2f_${arm}_s42_replay b2t_c2f_${arm}_s42 $arm $P050 $A050 42 --candidate-subset c2f --c2f-top-k 5
done
add XR.lat.o0 XR replay b3_lat_o0_s42_replay b3_lat_o0_s42 BF $S128 $A050 42 --frame-avg-angles 0
add XR.lat.o0_180 XR replay b3_lat_o0_180_s42_replay b3_lat_o0_180_s42 BF $S128 $A050 42 --frame-avg-angles 0 180
# ---- CPU summaries
B1RUNS=() ; for arm in P1 YAW BF; do for k in 8 1; do B1RUNS+=(--run $arm $k $D/runs/b1_k${k}_${arm}_s42); done; done
B1K8=() ; for arm in P1 YAW BF; do B1K8+=(--split $arm 8 $D/runs/b1_split_k8_${arm}_s42); done
B1K1=() ; for arm in P1 YAW BF; do B1K1+=(--split $arm 1 $D/runs/b1_split_k1_${arm}_s42); done
B1X=(--selection $S128 --selector $D/selector/selector_128.json --geometry-audit $A050)
add B1S.all B1S cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_summarize_latency.py ${B1X[*]} ${B1RUNS[*]} ${B1K8[*]} ${B1K1[*]} \
  --output-json $D/b1/b1_latency_summary.json --output-md $D/b1/b1_latency_summary.md
add B1S.noK1 B1S cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_summarize_latency.py ${B1X[*]} ${B1RUNS[*]} ${B1K8[*]} \
  --output-json $D/b1/b1_latency_summary_noK1split.json --output-md $D/b1/b1_latency_summary_noK1split.md
add AGG.spec AGG cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_build_b2b3_spec.py --inputs $D/b2b3/b2b3_inputs.json --runs-root $D/runs \
  --synthesis $D/b2b3/synthesis_cells.json --output $D/b2b3/b2b3_spec.json
add AGG.run AGG cpu env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=$D/b2b3/.mpl $PY tools/exp37_aggregate_b2b3.py --spec $D/b2b3/b2b3_spec.json \
  --output-json $D/b2b3/b2b3_summary.json --output-md $D/b2b3/b2b3_summary.md --figure-stem $D/b2b3/b2_pareto
add AGGP.spec AGGP cpu env CUDA_VISIBLE_DEVICES= $PY tools/exp37_build_b2b3_spec.py --inputs $D/b2b3/b2b3_inputs.json --runs-root $D/runs \
  --output $D/b2b3/b2b3_spec_pending.json
add AGGP.run AGGP cpu env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=$D/b2b3/.mpl $PY tools/exp37_aggregate_b2b3.py --spec $D/b2b3/b2b3_spec_pending.json \
  --output-json $D/b2b3/b2b3_pending.json --output-md $D/b2b3/b2b3_pending.md

# ---------------------------------------------------------------- the engine
LOCALIZE=(--model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --agree-ckpt weights/AGREE/AGREE_fullAR.pt
          --context-manifest $E9/context_manifest_exp01_seed42.json --dataset-root AcousticRooms
          --mesh-root AcousticRooms/room_mesh_obj_format --device cuda:0 --candidate-batch-size 64 --tau 0.1
          --score-sample-counts 1 --require-attention-backend sdpa)
TIMING=(--synchronize-latency --warmup-query-count 1 --measure-core-forward)

refuse() { echo "REFUSED [$1]: $2" >&2; return 2; }

# build_cmd STEP -> fills CMD (argv) and OUTS (paths that must not exist) and NEEDS (paths that must exist)
build_cmd() {
  local id=$1 kind run arm sel audit seed of; read -r -a a <<< "${ARGS[$id]}"; kind=${a[0]}
  CMD=() OUTS=() NEEDS=()
  case $kind in
    cpu) CMD=("${a[@]:1}") ;;
    gate) CMD=(env CUDA_VISIBLE_DEVICES= $PY tools/exp37_cotenancy_monitor.py --gate-only --output $D/gates/${a[1]}.json); OUTS=($D/gates/${a[1]}.json) ;;
    selector) CMD=(env CUDA_VISIBLE_DEVICES=0 $PY $L tools/exp37_selector_latency.py --selection ${a[2]}
                   --context-manifest $E9/context_manifest_exp01_seed42.json --dataset-root AcousticRooms
                   --agree-checkpoint weights/AGREE/AGREE_fullAR.pt --device cuda:0 --candidate-batch-size 64
                   --repeat-count 3 --output $D/selector/${a[1]}.json)
              OUTS=($D/selector/${a[1]}.json $D/selector/${a[1]}.json.cotenancy) ;;
    g0cmp) CMD=(env CUDA_VISIBLE_DEVICES= $PY tools/exp37_g0_compare.py --arm ${a[1]} --run-dir $D/runs/${a[2]}
                --output $D/g0/g0_${a[1]}_verdict.json)
           [ "${a[1]}" = YAW ] || CMD+=(--yaw-verdict $D/g0/g0_YAW_verdict.json)
           OUTS=($D/g0/g0_${a[1]}_verdict.json); NEEDS=($D/runs/${a[2]}/run_manifest.json) ;;
    timed|acc) run=${a[1]} arm=${a[2]} sel=${a[3]} audit=${a[4]} seed=${a[5]}
               CMD=(env CUDA_VISIBLE_DEVICES=0 $PY $L localize_FLAC.py --ckpt-path ${CKPT[$arm]} --cond-method ${COND[$arm]}
                    --pilot-manifest $sel --geometry-audit $audit --sample-seed $seed --output-dir $D/runs/$run
                    "${LOCALIZE[@]}" "${a[@]:6}")
               OUTS=($D/runs/$run)
               if [ $kind = timed ]; then CMD+=("${TIMING[@]}"); else CMD+=(--dump-waveforms $W/$run); OUTS+=($W/$run); fi ;;
    replay) run=${a[1]} of=${a[2]} arm=${a[3]} sel=${a[4]} audit=${a[5]} seed=${a[6]}
            CMD=(env CUDA_VISIBLE_DEVICES=0 $PY $L localize_FLAC.py --ckpt-path ${CKPT[$arm]} --cond-method ${COND[$arm]}
                 --pilot-manifest $sel --geometry-audit $audit --sample-seed $seed --output-dir $D/runs/$run
                 "${LOCALIZE[@]}" "${a[@]:7}" --dump-waveforms $W/$run --replay-of $D/runs/$of)
            OUTS=($D/runs/$run $W/$run); NEEDS=($D/runs/$of/run_manifest.json) ;;
    *) refuse "$id" "unknown step kind $kind"; return 2 ;;
  esac
}

run_step() {
  local id=$1 log rc path
  [ -n "${PHASE[$id]:-}" ] || { refuse "$id" "no such step (bash $0 list)"; return 2; }
  build_cmd "$id" || return 2
  cd "$REPO_DIR" || { refuse "$id" "no checkout at $REPO_DIR"; return 2; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { refuse "$id" "the tracked tree is dirty"; return 2; }
  for path in "${OUTS[@]}"; do [ ! -e "$path" ] || { refuse "$id" "$path exists (outputs are fresh)"; return 2; }; done
  for path in "${NEEDS[@]}"; do [ -e "$path" ] || { refuse "$id" "$path is missing"; return 2; }; done
  mkdir -p "$D"
  log=$D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_${id}.log
  { echo "HEAD $(git rev-parse HEAD) step=$id phase=${PHASE[$id]}"; printf 'COMMAND'; printf ' %q' "${CMD[@]}"; echo; } \
    | tee "$log"
  if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: not executed" | tee -a "$log"; return 0; fi
  "${CMD[@]}" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  echo "EXIT $rc" | tee -a "$log"
  return "$rc"
}

run_steps() {
  local id rc
  for id in "$@"; do
    run_step "$id"; rc=$?
    if [ "$rc" -ne 0 ]; then echo "STOPPED at $id (exit $rc)" >&2; return "$rc"; fi
  done
}

case ${1:-} in
  list) for id in "${IDS[@]}"; do build_cmd "$id"; printf '%-22s %-5s' "$id" "${PHASE[$id]}"; printf ' %q' "${CMD[@]}"; echo; done
        echo "B3S                    B3S   $SYNTHESIS_PENDING" ;;
  run) shift; [ "$#" -gt 0 ] || { echo "usage: bash $0 run STEP [STEP ...]" >&2; exit 2; }
       [ "$*" != B3S ] || { refuse B3S "an external step: $SYNTHESIS_PENDING"; exit 2; }
       run_steps "$@"; exit $? ;;
  phase) [ -n "${2:-}" ] || { echo "usage: bash $0 phase PHASE" >&2; exit 2; }
         sel=(); for id in "${IDS[@]}"; do [ "${PHASE[$id]}" = "$2" ] && sel+=("$id"); done
         [ "${#sel[@]}" -gt 0 ] || { refuse "$2" "no such phase"; exit 2; }
         run_steps "${sel[@]}"; exit $? ;;
  *) echo "usage: bash $0 list | run STEP [STEP ...] | phase PHASE" >&2; exit 2 ;;
esac
