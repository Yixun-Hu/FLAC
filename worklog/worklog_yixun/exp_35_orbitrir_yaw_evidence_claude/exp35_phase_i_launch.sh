#!/usr/bin/env bash
# exp_35 phase (i): the Planner's frozen launch entry and the supervisor launch (plan_orbitrir_yaw_evidence.md §7, §8,
# §12 B1-B3, §13 B1 / NIT 3, and the launch-entry lists of Codex code reviews C3c and C3d).
#
#   bash exp35_phase_i_launch.sh capture <H35>
#       Read-only, except for NEW files that are never overwritten: the approved co-tenant tuple
#       <run worktree>/<exp_35 folder>/exp35_approved_cotenants_<ts>.json (+ its record copy) and three logs in the main
#       checkout's exp_35 folder. Exit 0 only when every CHECK holds. Prints the launch line.
#   bash exp35_phase_i_launch.sh launch <H35> <tuple path> <tuple sha256>
#       Re-checks the launch preconditions, repeats the supervisor's read-only dry run in the EXACT launch environment,
#       then starts the supervisor detached (setsid + nohup, a minimal `env -i` environment, cwd = the run worktree) and
#       records its identity and first heartbeat. The supervisor's own flock refuses a second instance.
#
# Never signals, kills or edits any process; never writes into Yixun's checkout or the NAS; nvidia-smi queries only.
# Written by the exp_35 Planner (Claude Opus 5.5 main session).
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
LAUNCH_ENV=(HOME=/home/yixunhu USER=yixunhu LOGNAME=yixunhu SHELL=/bin/bash LANG=en_US.UTF-8
            PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
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
sha() { sha256sum < "$1" | cut -c1-64; }
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
procs_glob() {    # PIDs other than this script with an argv element whose basename matches the glob $1
  local d pid a; local -a argv
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}; [ "$pid" = "$$" ] && continue
    { mapfile -d '' argv < "$d/cmdline"; } 2>/dev/null || continue
    for a in "${argv[@]}"; do [[ "${a##*/}" == $1 ]] && { say "$pid"; break; }; done
  done
  return 0
}
apps() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | LC_ALL=C sort; }
args_for() {  # <tuple> <sha>: the supervisor's exact arguments
  SUP_ARGS=(--h35 "$H35" --approved "$1" --approved-sha256 "$2" --state "$STATE" --reservations "$RES"
            --his-log "$HIS_LOG" --logdir "$RUNX")
}

common_checks() {   # the preconditions both modes re-check right before they act
  local head; head="$(git -C "$RUN" rev-parse HEAD 2>/dev/null)"
  need "run worktree HEAD ${head} = H35" eq "$head" "$H35"
  need "branch exp-35-orbitrir-yaw" eq "$(git -C "$RUN" rev-parse --abbrev-ref HEAD 2>/dev/null)" exp-35-orbitrir-yaw
  need "the tracked tree is clean" eq "$(git -C "$RUN" status --porcelain --untracked-files=no 2>&1)" ""
  need "the compute apps are exactly his two s43 ranks (${RANK0} on GPU 0, ${TRAINER} on GPU 1)" \
    eq "$(apps)" "$(printf '%s\n' "${RANK0},${GPU0}" "${TRAINER},${GPU1}" | LC_ALL=C sort)"
  need "no exp35 supervisor is running" eq "$(procs_named exp35_supervisor.py)" ""
  need "no depth_canon_eval_run.sh is running" eq "$(procs_named depth_canon_eval_run.sh)" ""
  need "no exp_30 evaluation is running (his table job)" eq "$(procs_named exp30_attn_eval.sh)" ""
  need "the supervisor state file is absent (a new campaign)" test ! -e "$STATE"
  need "no reservation of GPU 0 or GPU 1 exists" test ! -e "$RES/$GPU0.json" -a ! -e "$RES/$GPU1.json"
}

dry_run() {   # <tuple> <sha>: the supervisor's read-only dry run in the exact launch environment; no file may change
  local log="$MAINX/orbitrir_yaw_evidence_${TS}_phase_i_supervisor_dry_run.log" mark rc
  mark="$(mktemp)"; sleep 1
  args_for "$1" "$2"
  ( cd "$RUN" && env -i "${LAUNCH_ENV[@]}" "$PY" "$SUP" --dry-run "${SUP_ARGS[@]}" ) > "$log" 2>&1; rc=$?
  cat "$log"
  say "dry run rc ${rc}; transcript ${log} sha256 $(sha "$log")"
  need "the supervisor dry run in the launch environment: rc 0 and DRY RUN OK" \
    test "$rc" = 0 -a "$(grep -c '^DRY RUN OK' "$log")" = 1
  need "the dry run wrote nothing under the run worktree (find -newer, .git excluded)" \
    eq "$(find "$RUN" -path "$RUN/.git" -prune -o -newer "$mark" -print 2>&1 | head -5)" ""
  need "the dry run created no reservation directory entry" test ! -e "$RES/$GPU0.json" -a ! -e "$RES/$GPU1.json"
  rm -f -- "$mark"
}

capture() {
  say "=== exp_35 phase (i) launch capture -- $(date -Is) -- host $(hostname) -- H35 ${H35} ==="
  say "this script: $(sha "$0")  $0"
  common_checks
  rec "git status of the run worktree (untracked shown)" git -C "$RUN" status --short
  rec "§8: the evaluator closure since d6ca307a" git -C "$RUN" diff --stat d6ca307a "$H35" -- eval_FLAC.py src ':(exclude)src/tests' data/AR
  need "§8: only src/localization/* changed since d6ca307a" eq "$(git -C "$RUN" diff --name-only d6ca307a "$H35" -- \
    eval_FLAC.py src ':(exclude)src/tests' data/AR | grep -v '^src/localization/')" ""

  say ""; say "### §7 assets (links created by the Planner at 18:08 EST Nov 5)"
  local p; for p in AcousticRooms weights/AGREE weights/FLAC/VAE.ckpt; do
    say "$p -> $(readlink "$RUN/$p") (resolves $(readlink -f "$RUN/$p"))"; done
  need "AcousticRooms -> the NAS" eq "$(readlink "$RUN/AcousticRooms")" /media/diskstation/yixunhu/FLAC/AcousticRooms
  need "weights/AGREE -> the main checkout" eq "$(readlink "$RUN/weights/AGREE")" "$MAIN/weights/AGREE"
  need "weights/FLAC/VAE.ckpt -> the main checkout" eq "$(readlink "$RUN/weights/FLAC/VAE.ckpt")" "$MAIN/weights/FLAC/VAE.ckpt"
  need "VAE.ckpt 6198ff0d…" eq "$(sha "$RUN/weights/FLAC/VAE.ckpt")" 6198ff0d781fa05f546d19a0137a830c3f265437193c3581d8043713e7271baf
  need "AGREE_fullAR.pt 3a13243d…" eq "$(sha "$RUN/weights/AGREE/AGREE_fullAR.pt")" 3a13243d6c6a11082697592c2c5db84790d37859451df2963eb51d655b23c787
  need "AcousticRooms/depth_map and metadata reachable" test -d "$RUN/AcousticRooms/depth_map" -a -d "$RUN/AcousticRooms/metadata"
  rec "the DINOv3 vits16 snapshot (HF_HUB_OFFLINE=1 at every cell)" bash -c \
    'for f in ~/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/*/*; do printf "%s  %s\n" "$(sha256sum < "$f" | cut -c1-64)" "$f"; done'

  rec "pinned files at H35 (bound by git; recorded)" bash -c 'cd "$1" && sha256sum \
    '"$X25"'/{depth_canon_cells.json,depth_canon_eval_run.sh,depth_canon_collect.py,depth_canon_cond_probe.py,depth_canon_make_cells.py,depth_canon_results.json,depth_canon_cond_probe.json,FLAC_AR_CANON.json} \
    '"$REL"'/{exp35_supervisor.py,exp35_training_guard.py,exp35_gpu_identity.py,exp35_exp25_stream_registry.json} \
    overlays/no_flash/{flash_attn.py,exp35_no_flash_launch.py} worklog/worklog_yixun/exp_07_fa_scratch_claude/{FLAC_AR_BVp1.json,FLAC_AR_BF.json} \
    src/configs/dataset_configs/AR/eval/{acousticroom_unseeneval.json,acousticroom_unseeneval_1.json} data/AR/unseen_eval.json \
    analysis/yaw_robustness/generated/yaw_metrics_raw.csv' _ "$RUN"
  need "cells.json 440eb1be…" eq "$(sha "$RUN/$X25/depth_canon_cells.json")" 440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d
  need "overlay 0d0e52ca…" eq "$(sha "$RUN/overlays/no_flash/flash_attn.py")" 0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023
  need "launcher bbd2847d…" eq "$(sha "$RUN/overlays/no_flash/exp35_no_flash_launch.py")" bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de
  need "exp_25 results ec3fcd08…" eq "$(sha "$RUN/$X25/depth_canon_results.json")" ec3fcd08e3d210524b50246d9bfdf084e72835432a5d06450085e0cad12eb6e0
  need "stream registry 9ec31f32…" eq "$(sha "$RUNX/exp35_exp25_stream_registry.json")" 9ec31f32a9d89cf9ffa90938d7f1198658af2eafa3b53675605d9fb925365a24
  need "A6 CSV 0ae554c0…" eq "$(sha "$RUN/analysis/yaw_robustness/generated/yaw_metrics_raw.csv")" 0ae554c09f45ce89b8957cf75a3d15e44f4173f4f8987497dbbebf16bbfb23da
  local cells_txt; cells_txt="$(mktemp)"
  rec "cells: rows, the 116 exp_35 rows, the exp_25-only render" bash -c 'cd "$1" && CUDA_VISIBLE_DEVICES= "$2" -B -c "
import hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
m = json.load(open(\"$3/depth_canon_cells.json\"))
rows = [r for r in m[\"cells\"] if r[\"order\"] >= 230]
print(\"rows\", len(m[\"cells\"]), \"exp35\", len(rows))
print(\"exp35_rows_sha256\", hashlib.sha256((\"\n\".join(json.dumps(r, separators=(\",\", \":\")) for r in rows) + \"\n\").encode()).hexdigest())
s = importlib.util.spec_from_file_location(\"mk\", \"$3/depth_canon_make_cells.py\"); mk = importlib.util.module_from_spec(s); s.loader.exec_module(mk)
out = mk.render(mk.build_manifest(exp35=False)); out = out.encode() if isinstance(out, str) else out
print(\"exp25_only_render_sha256\", hashlib.sha256(out).hexdigest())
" 2>/dev/null' _ "$RUN" "$PY" "$X25" | tee "$cells_txt"
  need "346 rows / 116 exp_35 rows" grep -qx 'rows 346 exp35 116' "$cells_txt"
  need "the 116 exp_35 rows 500d6f5e…" grep -qx 'exp35_rows_sha256 500d6f5ed9c70700fdd4c778895fc84e01f7756fbac84ad2471a9346ab9252ea' "$cells_txt"
  need "the exp_25-only render 1073ef64… (= d6ca307a's file)" grep -qx "exp25_only_render_sha256 $(git -C "$RUN" show d6ca307a:"$X25"/depth_canon_cells.json | sha256sum | cut -c1-64)" "$cells_txt"
  rm -f -- "$cells_txt"
  need "make_cells --check (the committed manifest has not drifted)" bash -c 'cd "$1" && CUDA_VISIBLE_DEVICES= "$2" -B "$3/depth_canon_make_cells.py" --check' _ "$RUN" "$PY" "$X25"
  rec "checkpoints: the NAS MANIFEST lines (the runner re-hashes and stages each source per cell)" \
    grep -n -E 'c4c678826cddda37|5319feb4af874624|52c64b52' /media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/MANIFEST.sha256
  need "the dump root exists" test -d /media/diskstation/yixunhu/FLAC/exp25_pred_waveforms

  rec "interpreter and packages" bash -c 'CUDA_VISIBLE_DEVICES= "$1" -c "import sys, torch, importlib.metadata as m; print(sys.version.split()[0], torch.__version__, torch.version.cuda, \"flash-attn\", m.version(\"flash-attn\"))"' _ "$PY"
  need "pip freeze 8b7a802e… (the plan's environment)" eq "$("$PY" -m pip freeze 2>/dev/null | sha256sum | cut -c1-64)" 8b7a802e65114c44bbf98206f0ccce6b960d48e5b094941496c3801cea1defb2

  rec "nvidia-smi" nvidia-smi
  rec "GPU map / memory" nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
  rec "compute apps" nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  need "GPU 1 free memory >= 10,240 MiB (§8)" test "$(nvidia-smi --id="$GPU1" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" -ge 10240
  rec "his s43 ranks" ps -o pid,ppid,lstart,etime,cmd -p "$RANK0,$TRAINER"
  for p in "$RANK0" "$TRAINER"; do say "pid $p: cwd $(readlink /proc/$p/cwd) fd1 $(readlink /proc/$p/fd/1) fd2 $(readlink /proc/$p/fd/2)"; done
  rec "his log" stat -c 'inode %i size %s mtime %y' "$HIS_LOG"
  rec "the guard now (the last 1,000 steps judged for the rate; baseline = step - 1000)" bash -c \
    'g="$("$1" -B "$2" --log "$3")"; s="$(printf "%s" "$g" | "$1" -c "import json,sys; print(json.load(sys.stdin)[\"step\"][\"global\"])")"; "$1" -B "$2" --log "$3" --baseline-step "$((s - 1000))"' _ "$PY" "$GUARD" "$HIS_LOG"
  rec "the predictor for chunk 1 (CTL K1, 17.6 min): (40,000 - step) x 3.5 s > 2 x 17.6 min + 30 min" bash -c \
    's="$("$1" -B "$2" --log "$3" | "$1" -c "import json,sys; print(json.load(sys.stdin)[\"step\"][\"global\"])")"; echo "step $s: earliest s43 end in $(( (40000 - s) * 35 / 10 )) s; needed > $(( 2 * 1056 + 1800 )) s"' _ "$PY" "$GUARD" "$HIS_LOG"

  rec "his queue, evaluator and admission helper (§13 NIT 3)" "$PY" -B "$SUP" --external-shas
  need "queue 9394ca90…" eq "$(sha "$KIT/exp30_attn_queue.sh")" 9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa
  need "evaluator 94fcc9e1…" eq "$(sha "$KIT/exp30_attn_eval.sh")" 94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c
  need "admission helper 8572bb03…" eq "$(sha "$KIT/exp30_attn_queue_state.py")" 8572bb0305bcd1f1f245a9d2b31ca9760ba7ed036bccf5bf612d2b2208b33907
  rec "s44 completion evidence (§13 B1)" grep -n -E $'^ACEattn\t44\t0\t|^EVAL_DONE ACEattn s44 ' "$QSTATUS" "$QLOG"
  need "s44 status line" grep -qx $'ACEattn\t44\t0\t/home/yixunhu/codespace/flac-fork-attn/outputs_FLAC/exp30_ACEattn_s44/epoch=8-step=40000.ckpt' "$QSTATUS"
  need "s44 EVAL_DONE line" grep -qx 'EVAL_DONE ACEattn s44 cells=40 2026-11-04T16:01:40-05:00' "$QLOG"

  rec "the exp_22 free-card watcher (never built; must be absent)" bash -c 'ls /home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_22_loc_meshgrid_claude | grep -i watch; true'
  need "no process runs an exp_22 watcher (*exp22*watch* / *meshgrid*watch* / *free_card*)" \
    eq "$(procs_glob '*exp22*watch*'; procs_glob '*meshgrid*watch*'; procs_glob '*free_card*')" ""
  rec "the reservation directory" ls -la "$RES"
  rec "the reservation schema the supervisor writes and authenticates" grep -n -E '^RES_KEYS|"holder": "exp35"' "$SUP"

  local tuple="$RUNX/exp35_approved_cotenants_${TS}.json" tsha
  say ""; say "### the approved co-tenant tuple (a new file: ${tuple})"
  ( set -o noclobber; "$PY" -B "$IDH" --snapshot --gpu-uuid "$GPU1" --pid "$TRAINER" > "$tuple" ) || { say "CHECK FAILED: the tuple snapshot"; FAIL=1; }
  tsha="$(sha "$tuple")"; cat "$tuple"; say "bytes $(stat -c %s "$tuple"); sha256 ${tsha}"
  need "the tuple validates against its sha256" "$PY" -B "$IDH" --validate "$tuple" --approved-sha256 "$tsha"
  need "the tuple is exactly his GPU 1 rank (pid, cwd, fd1 = fd2 = his log)" "$PY" -B -c '
import json, sys
d = json.load(open(sys.argv[1])); p = d["processes"]
ok = (d["gpu_uuid"] == sys.argv[2] and len(p) == 1 and p[0]["pid"] == int(sys.argv[3])
      and p[0]["cwd"] == "/home/yixunhu/codespace/flac-fork-attn" and p[0]["fd1"] == p[0]["fd2"] == sys.argv[4])
sys.exit(0 if ok else 1)' \
    "$tuple" "$GPU1" "$TRAINER" "$HIS_LOG"
  need "the live GPU 1 set equals the tuple" "$PY" -B "$IDH" --check --approved "$tuple" --approved-sha256 "$tsha" \
    --gpu-uuid "$GPU1" --pids "$(apps | awk -F, -v u="$GPU1" '$2 == u {print $1}' | paste -sd, -)"
  need "the tuple's record copy (a new file in the main checkout) is byte-identical" bash -c \
    'set -o noclobber; cat "$1" > "$2" && [ "$(sha256sum < "$2" | cut -c1-64)" = "$3" ]' _ "$tuple" "$MAINX/${tuple##*/}" "$tsha"

  say ""; say "### the supervisor's read-only dry run in the launch environment"
  dry_run "$tuple" "$tsha"
  say ""; say "### the launch environment (env -i): ${LAUNCH_ENV[*]}"
  args_for "$tuple" "$tsha"
  say "### the launch line"; printf 'bash %q launch %q %q %q\n' "$0" "$H35" "$tuple" "$tsha"
  say "### the supervisor command it will start"; printf '%q ' env -i "${LAUNCH_ENV[@]}" setsid nohup "$PY" "$SUP" "${SUP_ARGS[@]}"; printf '\n'
  [ "$FAIL" = 0 ] && { say "CAPTURE OK"; return 0; }
  say "CAPTURE REFUSED: at least one CHECK failed (see above)"; return 1
}

launch() {
  local tuple="${3:-}" tsha="${4:-}" suplog pid
  say "=== exp_35 phase (i) LAUNCH -- $(date -Is) -- H35 ${H35} -- tuple ${tuple} ${tsha} ==="
  say "this script: $(sha "$0")  $0"
  need "the tuple file exists" test -f "$tuple" -a ! -L "$tuple"
  need "the tuple's sha256 is the captured value" eq "$(sha "$tuple")" "$tsha"
  need "the tuple validates" "$PY" -B "$IDH" --validate "$tuple" --approved-sha256 "$tsha"
  common_checks
  need "the live GPU 1 set equals the tuple" "$PY" -B "$IDH" --check --approved "$tuple" --approved-sha256 "$tsha" \
    --gpu-uuid "$GPU1" --pids "$(apps | awk -F, -v u="$GPU1" '$2 == u {print $1}' | paste -sd, -)"
  dry_run "$tuple" "$tsha"
  [ "$FAIL" = 0 ] || { say "LAUNCH REFUSED: at least one CHECK failed (see above)"; return 1; }
  suplog="$RUNX/exp35_supervisor_${TS}.log"
  ( set -o noclobber; : > "$suplog" ) || { say "LAUNCH REFUSED: ${suplog} exists"; return 1; }
  args_for "$tuple" "$tsha"
  say "starting: cd ${RUN} && env -i ${LAUNCH_ENV[*]} setsid nohup ${PY} ${SUP} ${SUP_ARGS[*]} >> ${suplog} 2>&1 < /dev/null &"
  ( cd "$RUN" && exec env -i "${LAUNCH_ENV[@]}" /usr/bin/setsid /usr/bin/nohup "$PY" "$SUP" "${SUP_ARGS[@]}" \
      >> "$suplog" 2>&1 < /dev/null ) &
  pid=$!
  sleep 20
  local found; found="$(procs_named exp35_supervisor.py)"
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

case "$MODE" in capture|launch) ;; *) say "usage: $0 capture <H35> | launch <H35> <tuple> <sha256>"; exit 2 ;; esac
[[ "$H35" =~ ^[0-9a-f]{40}$ ]] || { say "H35 must be 40-hex"; exit 2; }
OUT="$MAINX/orbitrir_yaw_evidence_${TS}_phase_i_${MODE}.log"
( set -o noclobber; : > "$OUT" ) || { say "${OUT} exists"; exit 2; }
"$MODE" "$@" 2>&1 | tee -a "$OUT"; rc=${PIPESTATUS[0]}
say "rc ${rc}; log ${OUT}" | tee -a "$OUT"
exit "$rc"
