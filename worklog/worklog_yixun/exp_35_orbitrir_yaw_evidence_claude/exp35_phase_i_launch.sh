#!/usr/bin/env bash
# exp_35 phase (i): the Planner's frozen launch entry and the supervisor launch (plan_orbitrir_yaw_evidence.md §7, §8,
# §12 B1-B3, §13 B1 / NIT 3; the launch-entry lists of Codex code reviews C3c, C3d, C3e and INT).
#
#   EXPECT_SCRIPT_SHA256=<reviewed sha> bash exp35_phase_i_launch.sh capture <H35>
#       Read-only except for NEW files (never overwritten): the approved co-tenant tuple
#       <run worktree>/<exp_35 folder>/exp35_approved_cotenants_<ts>.json, its record copy, the capture log, the
#       dry-run transcript and -- ONLY when every CHECK passed -- the capture SEAL (all in the main checkout's exp_35
#       folder except the tuple). The seal binds H35, this script's sha256, the tuple and its copy, the capture log,
#       the campaign digest, every external pin and the four-rung ladder at H35. Prints the launch line on success only.
#   EXPECT_SCRIPT_SHA256=<reviewed sha> bash exp35_phase_i_launch.sh launch <H35> <seal> <seal sha256>
#       Refuses unless the seal verifies, is younger than SEAL_MAX_AGE_S, and every input it binds still holds; re-runs
#       the preconditions and the supervisor's read-only dry run in the EXACT launch environment; then starts the
#       supervisor detached (setsid + nohup, a minimal `env -i` environment, cwd = the run worktree) and records its
#       identity and first heartbeat. The supervisor's own flock refuses a second instance.
#
# Never signals, kills or edits any process; never writes into Yixun's checkout or the NAS; nvidia-smi queries only.
# Written by the exp_35 Planner (Claude Opus 5.5 main session); revised after Codex INT (capture-launch binding)
# and INT-L2 (hidden-aware DINOv3 set, canonical names, freshness at the last look, a new dry-run transcript).
set -uo pipefail

MODE="${1:-}"; H35="${2:-}"
RUN=/home/yixunhu/codespace/exp-35-orbitrir-yaw
MAIN=/home/yixunhu/codespace/FLAC
REL=worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude
RUNX="$RUN/$REL"; MAINX="$MAIN/$REL"; X25=worklog/worklog_yixun/exp_25_depth_canon_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
SUP="$RUNX/exp35_supervisor.py"; IDH="$RUNX/exp35_gpu_identity.py"; GUARD="$RUNX/exp35_training_guard.py"
KIT=/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn
HIS_LOG="$KIT/logs/train_exp30_ACEattn_s43_20261104T061255_741947.log"
QSTATUS="$KIT/logs/queue_20261101T233607_741947.status"; QLOG="$KIT/logs/queue_20261101T233607_741947.log"
GPU0=GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e; GPU1=GPU-6ac2604a-e84e-0615-1221-ccbed7c268da
RANK0=3027719; TRAINER=3028196                       # his s43 ranks (GPU 0, GPU 1); the tuple approves TRAINER only
RES=/home/yixunhu/codespace/.gpu_reservations
STATE="$RUNX/exp35_supervisor_state.json"
TUPLE_DIR="$RUNX"                                     # the canonical home of the approved tuple
LADDER_DIR="$RUNX"                                    # the four-rung ladder transcripts at H35 (untracked)
TS_RE='[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}'
NAS_MANIFEST=/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/MANIFEST.sha256
DUMP_ROOT=/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms
DINO=/home/yixunhu/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m
LAUNCH_ENV=(HOME=/home/yixunhu USER=yixunhu LOGNAME=yixunhu SHELL=/bin/bash LANG=en_US.UTF-8 PYTHONDONTWRITEBYTECODE=1
            PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
SEAL_MAX_AGE_S=10800                                 # a capture older than 3 h is stale: capture again
# --- the pins (plan §8, §13 NIT 3; the repo's DINOv3 pin = exp_07 assert_arm_configs.py VIT_REV / VIT_FILES) ------- #
PIN_DIGEST=da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1   # the campaign digest at H35 (ladder 3)
PINS=(   # name|path (absolute, or relative to the run worktree)|sha256
  "vae|$RUN/weights/FLAC/VAE.ckpt|6198ff0d781fa05f546d19a0137a830c3f265437193c3581d8043713e7271baf"
  "agree|$RUN/weights/AGREE/AGREE_fullAR.pt|3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787"
  "dino_config|$DINO/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32/config.json|9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b"
  "dino_weights|$DINO/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32/model.safetensors|4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
  "queue|$KIT/exp30_attn_queue.sh|9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa"
  "evaluator|$KIT/exp30_attn_eval.sh|94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c"
  "admission|$KIT/exp30_attn_queue_state.py|8572bb0305bcd1f1f245a9d2b31ca9760ba7ed036bccf5bf612d2b2208b33907"
  "cells|$RUN/$X25/depth_canon_cells.json|440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d"
  "overlay|$RUN/overlays/no_flash/flash_attn.py|0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023"
  "launcher|$RUN/overlays/no_flash/exp35_no_flash_launch.py|bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de"
  "exp25_results|$RUN/$X25/depth_canon_results.json|ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0"
  "stream_registry|$RUNX/exp35_exp25_stream_registry.json|9ec31f32a9d89cf9ffa90938d7f1198658af2eafa3b53675605d9fb925365a24"
  "a6_raw|$RUN/analysis/yaw_robustness/generated/yaw_metrics_raw.csv|0ae554c09f45ce89b8957cf75a3d15e44f4173f4f8987497dbbebf16bbfb23da"
  "a6_summary|$RUN/analysis/yaw_robustness/generated/yaw_metrics_summary.csv|e4602115f9e75479cdb76166b53a2907d11b8d32c64e0d61f1fa2d5342b09034"
  "a6_c4_raw|$RUN/analysis/yaw_robustness/generated/yaw_c4_metrics_raw.csv|441ade8b4931e69a358d760fb2b4fe29d72a6c40f4c00f5582eda0690a2598de"
  "a6_c4_summary|$RUN/analysis/yaw_robustness/generated/yaw_c4_metrics_summary.csv|b21ffbe7b264178bd7768bad8ac441c9dbd037217a0198c61c3ea5892cc9bf91"
)
PIN_PIP=8b7a802e65114c44bbf98206f0ccce6b960d48e5b094941496c3801cea1defb2       # `pip freeze | sha256sum` (plan §8)
PIN_VERSIONS="3.10.20 2.7.0+cu126 12.6 2.7.4.post1"                             # python, torch, CUDA, flash-attn
NAS_LINES=("c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c  P1/epoch=8-step=40000.ckpt"
           "5319feb4af874624859e87105ddd8ab06d4b449769d1e054f712b2b1c0542328  BF/epoch=8-step=40000.ckpt")
TS="$(date +%Y-%m-%d_%H-%M-%S)"
FAIL=0

say() { printf '%s\n' "$*"; }
rec() {   # <title> <command...>: the command, its output, its rc (recorded, never judged)
  say ""; say "### $1"; shift; printf '$'; printf ' %q' "$@"; printf '\n'; "$@" 2>&1; say "(rc $?)"
}
need() {  # <what> <command...>: a hard check
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then say "CHECK OK: $what"; else say "CHECK FAILED: $what"; FAIL=1; fi
}
sha() { sha256sum < "$1" 2>/dev/null | cut -c1-64; }
eq() { [ "$1" = "$2" ]; }
procs_named() {   # PIDs other than this script with an argv element whose basename is exactly $1 (no pgrep: no self-match)
  local d pid a; local -a argv
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}; [ "$pid" = "$$" ] && continue
    { mapfile -d '' argv < "$d/cmdline"; } 2>/dev/null || continue
    for a in "${argv[@]}"; do [ "${a##*/}" = "$1" ] && { say "$pid"; break; }; done
  done
  return 0
}
procs_glob() {    # the same, for a basename glob
  local d pid a; local -a argv
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}; [ "$pid" = "$$" ] && continue
    { mapfile -d '' argv < "$d/cmdline"; } 2>/dev/null || continue
    for a in "${argv[@]}"; do [[ "${a##*/}" == $1 ]] && { say "$pid"; break; }; done
  done
  return 0
}
apps() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | LC_ALL=C sort; }
gpu1_pids() { apps | awk -F, -v u="$GPU1" '$2 == u {print $1}' | paste -sd, -; }
args_for() {  # <tuple> <sha>: the supervisor's exact arguments
  SUP_ARGS=(--h35 "$H35" --approved "$1" --approved-sha256 "$2" --state "$STATE" --reservations "$RES"
            --his-log "$HIS_LOG" --logdir "$RUNX")
}
campaign_digest() {   # the runner's tree_digest for an exp_35 selection (depth_canon_eval_run.sh:136-150), recomputed
  ( cd "$RUN" || exit 1
    mapfile -t cfgs < <(jq -r '.arms[].model_config' "$X25/depth_canon_cells.json") || exit 1
    git ls-files -z --cached --others --exclude-standard -- eval_FLAC.py src ':(exclude)src/tests' data/AR "${cfgs[@]}" \
        "$X25/depth_canon_eval_run.sh" "$X25/depth_canon_collect.py" "$X25/depth_canon_make_cells.py" \
        "$X25/depth_canon_cells.json" "$X25/depth_canon_audit_gate.py" overlays/no_flash "$REL/exp35_gpu_identity.py" \
      | LC_ALL=C sort -zu | xargs -0r sha256sum -- | sha256sum | cut -c1-64 )
}
versions() { CUDA_VISIBLE_DEVICES= "$PY" -B -c 'import sys, torch, importlib.metadata as m
print(sys.version.split()[0], torch.__version__, torch.version.cuda, m.version("flash-attn"))' 2>/dev/null | tail -n 1; }

script_check() {
  need "EXPECT_SCRIPT_SHA256 is set and equals this script's sha256 (${SELF_SHA})" eq "${EXPECT_SCRIPT_SHA256:-unset}" "$SELF_SHA"
}
common_checks() {   # the preconditions both modes re-check right before they act
  local head; head="$(git -C "$RUN" rev-parse HEAD 2>/dev/null)"
  need "run worktree HEAD ${head} = H35" eq "$head" "$H35"
  need "branch exp-35-orbitrir-yaw" eq "$(git -C "$RUN" rev-parse --abbrev-ref HEAD 2>/dev/null)" exp-35-orbitrir-yaw
  need "the tracked tree is clean" eq "$(git -C "$RUN" status --porcelain --untracked-files=no 2>&1)" ""
  need "the campaign digest recomputes to ${PIN_DIGEST}" eq "$(campaign_digest)" "$PIN_DIGEST"
  need "the compute apps are exactly his two s43 ranks (${RANK0} on GPU 0, ${TRAINER} on GPU 1)" \
    eq "$(apps)" "$(printf '%s\n' "${RANK0},${GPU0}" "${TRAINER},${GPU1}" | LC_ALL=C sort)"
  need "GPU 1 free memory >= 10,240 MiB (§8)" test "$(nvidia-smi --id="$GPU1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" -ge 10240
  need "no exp35 supervisor is running" eq "$(procs_named exp35_supervisor.py)" ""
  need "no depth_canon_eval_run.sh is running" eq "$(procs_named depth_canon_eval_run.sh)" ""
  need "no exp_30 evaluation is running (his table job)" eq "$(procs_named exp30_attn_eval.sh)" ""
  need "no exp_22 watcher runs (*exp22*watch* / *meshgrid*watch* / *free_card*)" \
    eq "$(procs_glob '*exp22*watch*'; procs_glob '*meshgrid*watch*'; procs_glob '*free_card*')" ""
  need "no exp_22 watcher script exists in the exp_22 folder" \
    eq "$(ls "$MAIN/worklog/worklog_yixun/exp_22_loc_meshgrid_claude" 2>/dev/null | grep -i -E 'watch|free_card')" ""
  need "the supervisor state file and its lock are absent (a new campaign)" test ! -e "$STATE" -a ! -e "$STATE.lock"
  need "no reservation of GPU 0 or GPU 1 exists" test ! -e "$RES/$GPU0.json" -a ! -e "$RES/$GPU1.json"
}
mutable_checks() {   # every external / untracked input the run reads, re-hashed (both modes)
  local e name path want
  for e in "${PINS[@]}"; do
    IFS='|' read -r name path want <<<"$e"
    need "pin ${name}: ${path##*/} ${want:0:8}…" eq "$(sha "$path")" "$want"
  done
  need "the DINOv3 cache holds exactly snapshot 114c1379… with exactly config.json + model.safetensors (hidden entries counted)" \
    eq "$(find "$DINO/snapshots" -mindepth 1 -maxdepth 1 -printf '%f\n' 2>&1 | LC_ALL=C sort | paste -sd' ' -) | $(find "$DINO/snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32" -mindepth 1 -printf '%P\n' 2>&1 | LC_ALL=C sort | paste -sd' ' -)" \
       "114c1379950215c8b35dfcd4e90a5c251dde0d32 | config.json model.safetensors"
  need "AcousticRooms -> the NAS" eq "$(readlink "$RUN/AcousticRooms")" /media/diskstation/yixunhu/FLAC/AcousticRooms
  need "weights/AGREE -> the main checkout" eq "$(readlink "$RUN/weights/AGREE")" "$MAIN/weights/AGREE"
  need "weights/FLAC/VAE.ckpt -> the main checkout" eq "$(readlink "$RUN/weights/FLAC/VAE.ckpt")" "$MAIN/weights/FLAC/VAE.ckpt"
  need "AcousticRooms/depth_map and metadata reachable" test -d "$RUN/AcousticRooms/depth_map" -a -d "$RUN/AcousticRooms/metadata"
  need "the NAS MANIFEST lists P1 and BF at their pins" bash -c 'grep -qxF -- "$2" "$1" && grep -qxF -- "$3" "$1"' _ "$NAS_MANIFEST" "${NAS_LINES[@]}"
  need "the dump root exists" test -d "$DUMP_ROOT"
  need "python / torch / CUDA / flash-attn = ${PIN_VERSIONS}" eq "$(versions)" "$PIN_VERSIONS"
  need "pip freeze ${PIN_PIP:0:8}… (the plan's environment)" eq "$("$PY" -B -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)" "$PIN_PIP"
  need "s44 status line (§13 B1)" grep -qx $'ACEattn\t44\t0\t/home/yixunhu/codespace/flac-fork-attn/outputs_FLAC/exp30_ACEattn_s44/epoch=8-step=40000.ckpt' "$QSTATUS"
  need "s44 EVAL_DONE line (§13 B1)" grep -qx 'EVAL_DONE ACEattn s44 cells=40 2026-11-04T16:01:40-05:00' "$QLOG"
}
ladder_checks() {   # <draft>: the four-rung ladder at H35 (untracked transcripts in the run worktree), content + sha256
  local short="${H35:0:8}" k f n
  local -a pat=("1_exp25_byte_guard" "2_supervisor_live_dry_run" "3_runner_dry_run_phase_i" "4_all_suites_in_worktree")
  for k in 0 1 2 3; do
    n="$(ls "$LADDER_DIR"/orbitrir_yaw_evidence_*_LADDER_${pat[$k]}_at_${short}.log 2>/dev/null | wc -l)"
    need "exactly one ladder rung $((k + 1)) transcript at ${short}" eq "$n" 1
    f="$(ls "$LADDER_DIR"/orbitrir_yaw_evidence_*_LADDER_${pat[$k]}_at_${short}.log 2>/dev/null | head -n 1)"
    say "ladder rung $((k + 1)): ${f##*/} sha256 $(sha "$f")"
    printf 'ladder_%s %s %s\n' "$((k + 1))" "${f##*/}" "$(sha "$f")" >> "$1"
    case $k in
      0) need "rung 1: ec3fcd08… and the md identical" bash -c 'grep -q "sha256 of the re-collected JSON: ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0" "$1" && grep -q "md == the committed exp_25 md: identical" "$1"' _ "$f" ;;
      1) need "rung 2: DRY RUN OK and the worktree unchanged" bash -c 'grep -q "^DRY RUN OK" "$1" && grep -q "IDENTICAL" "$1"' _ "$f" ;;
      2) need "rung 3: runner rc 0 and the digest ${PIN_DIGEST:0:8}…" bash -c 'grep -q "runner rc 0" "$1" && grep -q "evaluator tree digest $2" "$1"' _ "$f" "$PIN_DIGEST" ;;
      3) need "rung 4: a pytest summary with passes and no failure or error" bash -c 'l="$(grep -E "^[0-9]+ passed" "$1" | tail -n 1)"; [ -n "$l" ] && ! printf "%s" "$l" | grep -q -E "failed|error"' _ "$f"
         rec "rung 4 summary" bash -c 'grep -E "^[0-9]+ passed" "$1" | tail -n 1' _ "$f" ;;
    esac
  done
}
dry_run() {   # <tuple> <sha>: the supervisor's read-only dry run in the exact launch environment; no file may change
  local log="$MAINX/orbitrir_yaw_evidence_${TS}_phase_i_${MODE}_supervisor_dry_run.log" mark rc
  ( set -o noclobber; : > "$log" ) 2>/dev/null || { say "CHECK FAILED: the dry-run transcript ${log} already exists"; FAIL=1; return; }
  mark="$(mktemp)"; sleep 1
  args_for "$1" "$2"
  ( cd "$RUN" && env -i "${LAUNCH_ENV[@]}" "$PY" "$SUP" --dry-run "${SUP_ARGS[@]}" ) >> "$log" 2>&1; rc=$?
  cat "$log"
  say "dry run rc ${rc}; transcript ${log} sha256 $(sha "$log")"
  need "the supervisor dry run in the launch environment: rc 0 and DRY RUN OK" \
    test "$rc" = 0 -a "$(grep -c '^DRY RUN OK' "$log")" = 1
  need "the dry run wrote nothing under the run worktree (find -newer, .git excluded)" \
    eq "$(find "$RUN" -path "$RUN/.git" -prune -o -newer "$mark" -print 2>&1 | head -5)" ""
  need "the dry run created no reservation, state or lock file" test ! -e "$RES/$GPU0.json" -a ! -e "$RES/$GPU1.json" -a ! -e "$STATE" -a ! -e "$STATE.lock"
  rm -f -- "$mark"
}

capture() {   # <draft file>
  local draft="$1"
  say "=== exp_35 phase (i) launch capture -- $(date -Is) -- host $(hostname) -- H35 ${H35} ==="
  say "this script: ${SELF_SHA}  $0"
  script_check
  common_checks
  mutable_checks
  ladder_checks "$draft"
  rec "git status of the run worktree (untracked shown)" git -C "$RUN" status --short
  rec "§8: the evaluator closure since d6ca307a" git -C "$RUN" diff --stat d6ca307a "$H35" -- eval_FLAC.py src ':(exclude)src/tests' data/AR
  need "§8: only src/localization/* changed since d6ca307a" eq "$(git -C "$RUN" diff --name-only d6ca307a "$H35" -- \
    eval_FLAC.py src ':(exclude)src/tests' data/AR | grep -v '^src/localization/')" ""
  local cells_txt; cells_txt="$(mktemp)"
  ( cd "$RUN" && CUDA_VISIBLE_DEVICES= "$PY" -B - "$X25" > "$cells_txt" 2>/dev/null ) <<'PYEOF'
import hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
x = sys.argv[1]
m = json.load(open(f"{x}/depth_canon_cells.json"))
rows = [r for r in m["cells"] if r["order"] >= 230]
print("rows", len(m["cells"]), "exp35", len(rows))
print("exp35_rows_sha256", hashlib.sha256(("\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n").encode()).hexdigest())
s = importlib.util.spec_from_file_location("mk", f"{x}/depth_canon_make_cells.py"); mk = importlib.util.module_from_spec(s); s.loader.exec_module(mk)
out = mk.render(mk.build_manifest(exp35=False)); out = out.encode() if isinstance(out, str) else out
print("exp25_only_render_sha256", hashlib.sha256(out).hexdigest())
PYEOF
  cat "$cells_txt"
  need "346 rows / 116 exp_35 rows" grep -qx 'rows 346 exp35 116' "$cells_txt"
  need "the 116 exp_35 rows 500d6f5e…" grep -qx 'exp35_rows_sha256 500d6f5ed9c70700fdd4c778895fc84e01f7756fbac84ad2471a9346ab9252ea' "$cells_txt"
  need "the exp_25-only render = d6ca307a's manifest (1073ef64…)" grep -qx "exp25_only_render_sha256 $(git -C "$RUN" show d6ca307a:"$X25"/depth_canon_cells.json | sha256sum | cut -c1-64)" "$cells_txt"
  rm -f -- "$cells_txt"
  need "make_cells --check (the committed manifest has not drifted)" bash -c 'cd "$1" && CUDA_VISIBLE_DEVICES= "$2" -B "$3/depth_canon_make_cells.py" --check' _ "$RUN" "$PY" "$X25"
  rec "versions" versions
  rec "nvidia-smi" nvidia-smi
  rec "GPU map / memory" nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
  rec "compute apps" nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  rec "his s43 ranks" ps -o pid,ppid,lstart,etime,cmd -p "$RANK0,$TRAINER"
  local p; for p in "$RANK0" "$TRAINER"; do say "pid $p: cwd $(readlink /proc/$p/cwd) fd1 $(readlink /proc/$p/fd/1) fd2 $(readlink /proc/$p/fd/2)"; done
  rec "his log" stat -c 'inode %i size %s mtime %y' "$HIS_LOG"
  local step; step="$("$PY" -B "$GUARD" --log "$HIS_LOG" 2>/dev/null | "$PY" -B -c 'import json,sys; print(json.load(sys.stdin)["step"]["global"])' 2>/dev/null)"
  need "the guard reads his global step (${step:-none})" bash -c '[[ "$1" =~ ^[0-9]+$ ]]' _ "$step"
  rec "the guard over the last 1,000 steps (his current rate; baseline = step - 1000)" "$PY" -B "$GUARD" --log "$HIS_LOG" --baseline-step "$(( ${step:-1000} - 1000 ))"
  say "the predictor for chunk 1 (CTL K1, 17.6 min): step ${step:-?}: earliest s43 end in $(( (40000 - ${step:-40000}) * 35 / 10 )) s; needed > $(( 2 * 1056 + 1800 )) s"
  rec "his queue, evaluator and admission helper (§13 NIT 3)" "$PY" -B "$SUP" --external-shas
  rec "s44 completion evidence (§13 B1)" grep -n -E $'^ACEattn\t44\t0\t|^EVAL_DONE ACEattn s44 ' "$QSTATUS" "$QLOG"
  rec "the reservation directory" ls -la "$RES"
  rec "the reservation schema the supervisor writes and authenticates" grep -n -E '^RES_KEYS|"holder": "exp35"' "$SUP"
  if [ "$FAIL" != 0 ]; then say "CAPTURE REFUSED before the tuple: at least one CHECK failed (see above)"; return 1; fi

  local tuple="$TUPLE_DIR/exp35_approved_cotenants_${TS}.json" copy="$MAINX/exp35_approved_cotenants_${TS}.json" tsha
  say ""; say "### the approved co-tenant tuple (a new file: ${tuple})"
  ( set -o noclobber; "$PY" -B "$IDH" --snapshot --gpu-uuid "$GPU1" --pid "$TRAINER" > "$tuple" ) || { say "CHECK FAILED: the tuple snapshot"; FAIL=1; }
  tsha="$(sha "$tuple")"; cat "$tuple"; say "bytes $(stat -c %s "$tuple" 2>/dev/null); sha256 ${tsha}"
  need "the tuple validates against its sha256" "$PY" -B "$IDH" --validate "$tuple" --approved-sha256 "$tsha"
  need "the tuple is exactly his GPU 1 rank (pid, cwd, fd1 = fd2 = his log)" "$PY" -B -c '
import json, sys
d = json.load(open(sys.argv[1])); p = d["processes"]
ok = (d["gpu_uuid"] == sys.argv[2] and len(p) == 1 and p[0]["pid"] == int(sys.argv[3])
      and p[0]["cwd"] == "/home/yixunhu/codespace/flac-fork-attn" and p[0]["fd1"] == p[0]["fd2"] == sys.argv[4])
sys.exit(0 if ok else 1)' "$tuple" "$GPU1" "$TRAINER" "$HIS_LOG"
  need "the live GPU 1 set equals the tuple" "$PY" -B "$IDH" --check --approved "$tuple" --approved-sha256 "$tsha" --gpu-uuid "$GPU1" --pids "$(gpu1_pids)"
  say ""; say "### the supervisor's read-only dry run in the launch environment"
  dry_run "$tuple" "$tsha"
  if [ "$FAIL" != 0 ]; then say "CAPTURE REFUSED: at least one CHECK failed (see above); no record copy, no seal"; return 1; fi
  need "the tuple's record copy (a new file in the main checkout) is byte-identical" bash -c \
    'set -o noclobber; cat "$1" > "$2" && [ "$(sha256sum < "$2" | cut -c1-64)" = "$3" ]' _ "$tuple" "$copy" "$tsha"
  [ "$FAIL" = 0 ] || { say "CAPTURE REFUSED: the record copy"; return 1; }
  { printf 'h35 %s\nscript_sha256 %s\ntuple %s %s\ntuple_copy %s %s\ncampaign_digest %s\n' "$H35" "$SELF_SHA" \
      "$tuple" "$tsha" "$copy" "$(sha "$copy")" "$PIN_DIGEST"
    local e name path want; for e in "${PINS[@]}"; do IFS='|' read -r name path want <<<"$e"; printf 'pin_%s %s %s\n' "$name" "$path" "$want"; done
    printf 'pip_freeze %s\nversions %s\nhis_step %s\n' "$PIN_PIP" "$PIN_VERSIONS" "$step"; } >> "$draft"
  say "CAPTURE CHECKS OK (the seal is written after this log is closed)"
  return 0
}

seal_write() {   # <draft> <capture log>: the seal, a NEW file, only after a fully passing capture
  local seal="$MAINX/orbitrir_yaw_evidence_${TS}_phase_i_capture_seal.txt"
  ( set -o noclobber; { cat "$1"; printf 'capture_log %s %s\nsealed_at %s %s\n' "$2" "$(sha "$2")" "$(date +%s)" "$(date -Is)"; } > "$seal" ) \
    || { say "SEAL REFUSED: ${seal} exists"; return 1; }
  say "SEAL ${seal} sha256 $(sha "$seal")"
  say "LAUNCH LINE: EXPECT_SCRIPT_SHA256=${SELF_SHA} bash $0 launch ${H35} ${seal} $(sha "$seal")"
}

launch() {
  local seal="${3:-}" ssha="${4:-}" tuple tsha copy csha clog clsha age suplog pid found
  say "=== exp_35 phase (i) LAUNCH -- $(date -Is) -- H35 ${H35} -- seal ${seal} ${ssha} ==="
  say "this script: ${SELF_SHA}  $0"
  script_check
  need "the seal is a regular file in the main checkout's exp_35 folder" bash -c '[ -f "$1" ] && [ ! -L "$1" ] && [ "$(dirname "$1")" = "$2" ]' _ "$seal" "$MAINX"
  need "the seal's sha256 is the given value" eq "$(sha "$seal")" "$ssha"
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED: the seal"; return 1; }
  cat "$seal"
  sv() { awk -v k="$1" '$1 == k {$1 = ""; sub(/^ /, ""); print; exit}' "$seal"; }
  need "the seal names this H35" eq "$(sv h35)" "$H35"
  need "the seal names this script" eq "$(sv script_sha256)" "$SELF_SHA"
  need "the seal names the pinned campaign digest" eq "$(sv campaign_digest)" "$PIN_DIGEST"
  read -r tuple tsha <<<"$(sv tuple)"; read -r copy csha <<<"$(sv tuple_copy)"; read -r clog clsha <<<"$(sv capture_log)"
  need "the tuple is the canonical run-worktree file" bash -c '[ "${1%/*}" = "$2" ] && [[ "${1##*/}" =~ ^exp35_approved_cotenants_$3\.json$ ]] && [ -f "$1" ] && [ ! -L "$1" ]' _ "$tuple" "$TUPLE_DIR" "$TS_RE"
  need "the tuple still hashes to the sealed value" eq "$(sha "$tuple")" "$tsha"
  need "the record copy is the main-checkout file of the same name, byte-identical" bash -c '[ "$1" = "$2/${3##*/}" ] && [ -f "$1" ] && [ ! -L "$1" ] && [ "$(sha256sum < "$1" | cut -c1-64)" = "$4" ] && [ "$4" = "$5" ]' _ "$copy" "$MAINX" "$tuple" "$csha" "$tsha"
  need "the capture log is the canonical main-checkout file" bash -c '[ "${1%/*}" = "$2" ] && [[ "${1##*/}" =~ ^orbitrir_yaw_evidence_$3_phase_i_capture\.log$ ]] && [ -f "$1" ] && [ ! -L "$1" ]' _ "$clog" "$MAINX" "$TS_RE"
  need "the capture log still hashes to the sealed value and shows CAPTURE CHECKS OK" bash -c '[ "$(sha256sum < "$1" | cut -c1-64)" = "$2" ] && grep -q "^CAPTURE CHECKS OK" "$1" && ! grep -q "^CHECK FAILED" "$1"' _ "$clog" "$clsha"
  local sealed; sealed="$(sv sealed_at | awk '{print $1}')"
  need "the seal's sealed_at is a decimal epoch (${sealed})" bash -c '[[ "$1" =~ ^[0-9]{10}$ ]]' _ "$sealed"
  [[ "$sealed" =~ ^[0-9]{10}$ ]] || sealed=0
  age=$(( $(date +%s) - sealed ))
  need "the seal is current (${age} s old <= ${SEAL_MAX_AGE_S} s)" test "$age" -ge 0 -a "$age" -le "$SEAL_MAX_AGE_S"
  local k name path want; for k in 1 2 3 4; do
    read -r name want <<<"$(sv "ladder_$k")"
    need "ladder rung $k: a canonical basename" bash -c '[[ "$1" =~ ^orbitrir_yaw_evidence_$2_LADDER_$3_[a-z0-9_]+_at_$4\.log$ ]]' _ "$name" "$TS_RE" "$k" "${H35:0:8}"
    need "ladder rung $k transcript unchanged since the capture" eq "$(sha "$LADDER_DIR/$name")" "$want"; done
  need "the tuple validates" "$PY" -B "$IDH" --validate "$tuple" --approved-sha256 "$tsha"
  mutable_checks
  common_checks
  need "the live GPU 1 set equals the tuple" "$PY" -B "$IDH" --check --approved "$tuple" --approved-sha256 "$tsha" --gpu-uuid "$GPU1" --pids "$(gpu1_pids)"
  dry_run "$tuple" "$tsha"
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED: at least one CHECK failed (see above)"; return 1; }
  # the last look, immediately before the detachment
  need "last look: HEAD, apps, state, lock, reservations" bash -c '[ "$(git -C "$1" rev-parse HEAD)" = "$2" ] && [ ! -e "$3" ] && [ ! -e "$3.lock" ] && [ ! -e "$4" ] && [ ! -e "$5" ]' _ "$RUN" "$H35" "$STATE" "$RES/$GPU0.json" "$RES/$GPU1.json"
  need "last look: the compute apps are his two ranks" eq "$(apps)" "$(printf '%s\n' "${RANK0},${GPU0}" "${TRAINER},${GPU1}" | LC_ALL=C sort)"
  age=$(( $(date +%s) - sealed ))
  need "last look: the seal is still current (${age} s old <= ${SEAL_MAX_AGE_S} s)" test "$sealed" -gt 0 -a "$age" -ge 0 -a "$age" -le "$SEAL_MAX_AGE_S"
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED at the last look"; return 1; }
  suplog="$RUNX/exp35_supervisor_${TS}.log"
  ( set -o noclobber; : > "$suplog" ) || { say "LAUNCH REFUSED: ${suplog} exists"; return 1; }
  args_for "$tuple" "$tsha"
  say "starting: cd ${RUN} && env -i ${LAUNCH_ENV[*]} setsid nohup ${PY} ${SUP} ${SUP_ARGS[*]} >> ${suplog} 2>&1 < /dev/null &"
  ( cd "$RUN" && exec env -i "${LAUNCH_ENV[@]}" /usr/bin/setsid /usr/bin/nohup "$PY" "$SUP" "${SUP_ARGS[@]}" \
      >> "$suplog" 2>&1 < /dev/null ) &
  pid=$!
  sleep 20
  found="$(procs_named exp35_supervisor.py)"
  say "background pid ${pid}; exp35_supervisor.py processes now: ${found:-none}"
  [ "$found" = "$pid" ] || { say "LAUNCH NOT CONFIRMED: the supervisor is not exactly pid ${pid} (inspect; never signalled here)"; cat "$suplog"; return 1; }
  say "supervisor pid ${pid}: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)"
  say "  cwd $(readlink /proc/$pid/cwd 2>/dev/null); start time $(awk '{print $22}' /proc/$pid/stat 2>/dev/null); sid $(ps -o sid= -p "$pid" 2>/dev/null)"
  say "  environment: $(tr '\0' ' ' < /proc/$pid/environ 2>/dev/null)"
  say "--- the supervisor log so far (${suplog}) ---"; cat "$suplog"; say "---"
  if [ -d "/proc/$pid" ] && grep -q 'supervisor started (a new state)' "$suplog" && ! grep -q 'REFUSED\|STOP LATCH' "$suplog" \
     && [ -f "$STATE" ]; then
    say "LAUNCHED: supervisor pid ${pid}, log ${suplog}, state ${STATE}"; return 0
  fi
  say "LAUNCH NOT CONFIRMED: inspect ${suplog} (the supervisor is never signalled by this script)"; return 1
}

SELF_SHA="$(sha "$0")"
case "$MODE" in capture|launch) ;; *) say "usage: EXPECT_SCRIPT_SHA256=<sha> $0 capture <H35> | launch <H35> <seal> <seal sha256>"; exit 2 ;; esac
[[ "$H35" =~ ^[0-9a-f]{40}$ ]] || { say "H35 must be 40-hex"; exit 2; }
OUT="$MAINX/orbitrir_yaw_evidence_${TS}_phase_i_${MODE}.log"
( set -o noclobber; : > "$OUT" ) || { say "${OUT} exists"; exit 2; }
DRAFT="$(mktemp)"
if [ "$MODE" = capture ]; then capture "$DRAFT" 2>&1 | tee -a "$OUT"; rc=${PIPESTATUS[0]}
else launch "$@" 2>&1 | tee -a "$OUT"; rc=${PIPESTATUS[0]}; fi
say "rc ${rc}; log ${OUT} sha256 $(sha "$OUT")"
if [ "$MODE" = capture ] && [ "$rc" = 0 ] && ! grep -q '^CHECK FAILED' "$OUT"; then seal_write "$DRAFT" "$OUT" || rc=1; fi
rm -f -- "$DRAFT"
exit "$rc"
