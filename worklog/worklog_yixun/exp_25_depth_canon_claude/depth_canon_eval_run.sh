#!/usr/bin/env bash
# ============================================================================
# depth_canon_eval_run.sh -- exp_25 depth_canon: the evaluation campaign driver
# (plan_depth_canon.md v9 §5 "Drivers:" + atomic cell completion), adapted from exp_17's
# yaw_aug_a6000_extras_run.sh with its review findings applied (explicit protocol flags,
# exact-filename completion, validated records, local farm, per-PID waits, one lock).
#
# Every cell is a row of depth_canon_cells.json; its argv and every artifact path come
# from `depth_canon_collect.py --plan` (one definition, eval_FLAC's own build_output_paths).
# Per cell, in the manifest's priority order:
#   checkpoint  exists; sha256 == the manifest's pin (P1/YAW: also listed in the NAS
#               MANIFEST.sha256; CANON: the launcher's checkpoint manifest); hashed once.
#   farm link   <farm>/<ARM>/<checkpoint basename> -> the checkpoint (a stale entry aborts).
#   predicate   (cycle 4b) SKIP only a complete-and-verified cell; QUARANTINE the rest.
#   run         one lane per GPU: >= 10 GiB free AND no compute process on the card (uuid-mapped,
#               fail-closed queries, re-checked right before the spawn); never signals anything.
#
# Usage:  EXPECT_HEAD=<40-hex pin> bash depth_canon_eval_run.sh [--blocks B,..]
#           [--only CELL,..] [--limit N] [--no-dump-blocks B,..] [--dump-root DIR]
#           [--lanes 0,1] [--canon-ckpt-manifest PATH]
#   DRY_RUN=1  every gate except the side-effecting / GPU ones: no link, no mkdir, no
#              quarantine, no evaluation; prints each cell's exact argv.
#   GPU_POLL_SECONDS (default 600), GPU_MAX_POLLS (default 0 = wait for ever).
#   ALLOW_SHARED_GPU=1  Yixun's explicit opt-in (decision A6): a card holding foreign compute
#              processes may be used; the memory floor and the query-integrity checks stay.
#   DEPTH_CANON_EVAL_HARNESS=<dir under /tmp holding a .depth_canon_eval_harness marker>:
#              test mode only -- DEPTH_CANON_EVAL_PY (stub evaluator) and DEPTH_CANON_CELLS
#              are honoured, and the farm / logs / lock move into that dir.
# Pin: EXPECT_HEAD == HEAD and a committed relevant tree at start; its content digest is
#   re-computed before EVERY spawn and after every cell (bound into the receipt): a change
#   stops the campaign (no further spawn; running cells finish; nothing is killed).
# Exit: 0 all selected cells complete (or DRY_RUN) | 1 a cell failed | 2 a gate refused |
#   4 PIN DRIFT: HEAD or the relevant tree changed during the campaign.
#
# Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9 §5.
# ============================================================================
set -uo pipefail

REPO="$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
cd "$REPO" || exit 2
EXPDIR="worklog/worklog_yixun/exp_25_depth_canon_claude"
SELF="$EXPDIR/depth_canon_eval_run.sh"
COLLECT="$EXPDIR/depth_canon_collect.py"
MAKE_CELLS="$EXPDIR/depth_canon_make_cells.py"
CELLS="$EXPDIR/depth_canon_cells.json"
PY="/home/yixunhu/miniconda3/envs/flac/bin/python"        # the flac env, required
EVAL_PROG="eval_FLAC.py"
FARM_ROOT="outputs_FLAC/exp25_depth_canon"
LOGDIR="$EXPDIR"; LOCKDIR="$EXPDIR"
NAS_FSTYPE="cifs"; NAS_SOURCE="//10.249.7.28/shared_data"
MIN_GPU_FREE_MIB=10240
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-600}"; GPU_MAX_POLLS="${GPU_MAX_POLLS:-0}"
DRY_RUN="${DRY_RUN:-0}"; EXPECT_HEAD="${EXPECT_HEAD:-}"; HARNESS="${DEPTH_CANON_EVAL_HARNESS:-}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"

die() { echo "$*"; exit 2; }

BLOCKS=""; ONLY=""; LIMIT=""; NO_DUMP=""; DUMP_ROOT=""; LANES_CSV="0,1"; CANON_MANIFEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --blocks|--only|--limit|--no-dump-blocks|--dump-root|--lanes|--canon-ckpt-manifest)
      [ $# -ge 2 ] || die "$1 needs a value - abort"
      case "$1" in
        --blocks) BLOCKS="$2" ;; --only) ONLY="$2" ;; --limit) LIMIT="$2" ;;
        --no-dump-blocks) NO_DUMP="$2" ;; --dump-root) DUMP_ROOT="$2" ;;
        --lanes) LANES_CSV="$2" ;; --canon-ckpt-manifest) CANON_MANIFEST="$2" ;;
      esac
      shift 2 ;;
    *) die "unknown argument '$1' - abort" ;;
  esac
done
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1 (got '${DRY_RUN}') - abort" ;; esac
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
case "$ALLOW_SHARED_GPU" in 0|1) ;; *) die "ALLOW_SHARED_GPU must be 0 or 1 (got '${ALLOW_SHARED_GPU}') - abort" ;; esac
[[ "$LANES_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--lanes must be a comma list of GPU indices (got '${LANES_CSV}') - abort"
IFS=, read -r -a LANES <<<"$LANES_CSV"
[ "$(printf '%s\n' "${LANES[@]}" | sort -u | wc -l)" -eq "${#LANES[@]}" ] || die "--lanes repeats a GPU - abort"
[ -z "$LIMIT" ] || [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive integer - abort"
[[ "$GPU_POLL_SECONDS" =~ ^[0-9]+$ && "$GPU_MAX_POLLS" =~ ^[0-9]+$ ]] || die "GPU_POLL_SECONDS / GPU_MAX_POLLS must be integers - abort"

# --- harness (tests only): stub evaluator + manifest, everything under one /tmp dir -- #
HARNESS_FLAG=false; HREAL=""
if [ -n "$HARNESS" ]; then
  HREAL="$(readlink -f -- "$HARNESS" 2>/dev/null || true)"
  case "$HREAL" in /tmp/?*) ;; *) die "DEPTH_CANON_EVAL_HARNESS must be a directory under /tmp (got '${HARNESS}') - abort" ;; esac
  [ -d "$HREAL" ] && [ -f "$HREAL/.depth_canon_eval_harness" ] || die "DEPTH_CANON_EVAL_HARNESS '${HARNESS}' lacks its marker - abort"
  in_harness() { case "$(readlink -f -- "$1" 2>/dev/null)" in "$HREAL"/?*) [ -f "$1" ] ;; *) return 1 ;; esac; }
  EVAL_PROG="${DEPTH_CANON_EVAL_PY:-}"; CELLS="${DEPTH_CANON_CELLS:-}"
  in_harness "$EVAL_PROG" && in_harness "$CELLS" || die "HARNESS: DEPTH_CANON_EVAL_PY and DEPTH_CANON_CELLS must be files in ${HREAL} - abort"
  FARM_ROOT="$HREAL/farm"; LOGDIR="$HREAL/logs"; LOCKDIR="$HREAL"; HARNESS_FLAG=true
  mkdir -p "$LOGDIR" || die "HARNESS: cannot create ${LOGDIR} - abort"
fi

# --- one campaign at a time; everything teed into a timestamped log ------------------ #
exec 9>"$LOCKDIR/.depth_canon_eval.lock" || die "cannot open ${LOCKDIR}/.depth_canon_eval.lock - abort"
flock -n 9 || die "another evaluation campaign holds ${LOCKDIR}/.depth_canon_eval.lock - abort"
LOG="$LOGDIR/depth_canon_${TS}_eval_run.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_25 depth_canon eval campaign -- ${TS} -- HEAD $(git rev-parse HEAD) -- dry_run=${DRY_RUN} -- harness=${HARNESS_FLAG} ==="
[ -x "$PY" ] || die "the flac interpreter ${PY} is missing - abort"
for tool in jq flock findmnt sha256sum; do command -v "$tool" >/dev/null || die "${tool} is required - abort"; done
[ -f "$CELLS" ] || die "the cells manifest ${CELLS} is missing - abort"

# --- evaluator pin: HEAD + a committed evaluator / data / config / kit tree ---------- #
[[ "$EXPECT_HEAD" =~ ^[0-9a-f]{40}$ ]] || die "EXPECT_HEAD must be the full 40-hex evaluator pin (got '${EXPECT_HEAD}') - abort"
HEAD_SHA="$(git rev-parse HEAD)"
[ "$HEAD_SHA" = "$EXPECT_HEAD" ] || die "HEAD PIN FAILED: HEAD ${HEAD_SHA} != EXPECT_HEAD ${EXPECT_HEAD} - abort"
mapfile -t MODEL_CFGS < <(jq -r '.arms[].model_config' "$CELLS")
RELEVANT=(eval_FLAC.py src ':(exclude)src/tests' data/AR "${MODEL_CFGS[@]}" "$SELF" "$COLLECT" "$MAKE_CELLS"
          "$EXPDIR/depth_canon_cells.json" "$EXPDIR/depth_canon_audit_gate.py")
tree_dirty() { git status --porcelain --untracked-files=all -- "${RELEVANT[@]}"; }
tree_digest() {   # sha256 of the sorted sha256sum lines of every tracked + untracked (unignored) relevant file
  local lines
  lines="$(git ls-files -z --cached --others --exclude-standard -- "${RELEVANT[@]}" | LC_ALL=C sort -zu \
    | xargs -0r sha256sum --)" && [ -n "$lines" ] || return 1
  printf '%s\n' "$lines" | sha256sum | awk '{print $1}'
}
DIRTY="$(tree_dirty)"
[ -z "$DIRTY" ] || { echo "EVALUATOR TREE NOT COMMITTED at HEAD (every cell must run the pinned code):"; echo "$DIRTY"; exit 2; }
CAMPAIGN_DIGEST="$(tree_digest)" || die "TREE DIGEST FAILED: cannot hash the relevant tree - abort"
echo "evaluator pin OK: ${HEAD_SHA}; evaluator tree digest ${CAMPAIGN_DIGEST}"
while read -r path sha; do
  [ -f "$path" ] || die "CONFIG MISSING: ${path} - abort"
  got="$(sha256sum "$path" | awk '{print $1}')"
  [ "$got" = "$sha" ] || die "CONFIG PIN FAILED: ${path} is ${got}, the manifest pins ${sha} - abort"
done < <(jq -r '[.cells[] | (.model_config + " " + .model_config_sha256),
                 (.dataset_config + " " + .dataset_config_sha256)] | unique[]' "$CELLS")
echo "configs OK: every model / dataset config matches its manifest pin"
if [ "$HARNESS_FLAG" = false ]; then
  "$PY" "$MAKE_CELLS" --check || die "CELLS MANIFEST DRIFT: ${CELLS} is not the generator's output - abort"
fi

# --- the plan: selection, checkpoints, paths, argv (depth_canon_collect.py) --------- #
PLAN="$LOGDIR/depth_canon_${TS}_eval_plan.json"
PLAN_ARGS=(--plan "$PLAN" --cells "$CELLS" --farm-root "$FARM_ROOT" --no-dump-blocks "$NO_DUMP"
           --campaign-digest "$CAMPAIGN_DIGEST")
[ -n "$DUMP_ROOT" ] && PLAN_ARGS+=(--dump-root "$DUMP_ROOT")
[ -n "$BLOCKS" ] && PLAN_ARGS+=(--blocks "$BLOCKS")
[ -n "$ONLY" ] && PLAN_ARGS+=(--only "$ONLY")
[ -n "$LIMIT" ] && PLAN_ARGS+=(--limit "$LIMIT")
[ -n "$CANON_MANIFEST" ] && PLAN_ARGS+=(--canon-ckpt-manifest "$CANON_MANIFEST")
"$PY" "$COLLECT" "${PLAN_ARGS[@]}" || die "PLAN REFUSED (see above) - abort"
DUMP_ROOT="$(jq -r .dump_root "$PLAN")"
mapfile -t CELL_IDS < <(jq -r '.cells[].cell' "$PLAN")
echo "plan: ${#CELL_IDS[@]} cell(s); dump root ${DUMP_ROOT}; farm ${FARM_ROOT}; lanes ${LANES_CSV}"

# --- the dump root, verified ONCE: on the NAS share, created, write-probed ----------- #
nas_identity_ok() {
  local line
  line="$(findmnt -n -T "$1" -o FSTYPE,SOURCE 2>/dev/null || true)"
  grep -qE "^${NAS_FSTYPE}[[:space:]]+${NAS_SOURCE//./\\.}\$" <<<"$line" && return 0
  echo "DUMP ROOT GATE FAILED: ${1} is on '$(tr '\n' ';' <<<"$line")', not '${NAS_FSTYPE} ${NAS_SOURCE}' - abort"
  return 1
}
if [ "$(jq '[.cells[] | select(.dump)] | length' "$PLAN")" -gt 0 ]; then
  [[ "$DUMP_ROOT" = /?* && "$DUMP_ROOT" != *.staging-* ]] || die "DUMP ROOT GATE FAILED: '${DUMP_ROOT}' must be absolute, no '.staging-' - abort"
  [ ! -L "$DUMP_ROOT" ] || die "DUMP ROOT GATE FAILED: ${DUMP_ROOT} is a symlink - abort"
  anc="$DUMP_ROOT"; while [ ! -e "$anc" ] && [ "$anc" != / ]; do anc="$(dirname "$anc")"; done
  nas_identity_ok "$anc" || exit 2
  if [ "$DRY_RUN" = 1 ]; then
    echo "dump root OK under DRY_RUN: ${DUMP_ROOT} ($([ -d "$DUMP_ROOT" ] && echo exists || echo 'would be created'))"
  else
    mkdir -p "$DUMP_ROOT" && nas_identity_ok "$DUMP_ROOT" || die "DUMP ROOT GATE FAILED: cannot create ${DUMP_ROOT} - abort"
    probe="$DUMP_ROOT/.depth_canon_eval_probe_$$"
    { printf 'probe\n' > "$probe" && [ "$(cat "$probe")" = probe ] && rm -f "$probe" && [ ! -e "$probe" ]; } 2>/dev/null \
      || { rm -f "$probe" 2>/dev/null; die "DUMP ROOT GATE FAILED: write probe in ${DUMP_ROOT} - abort"; }
    echo "dump root OK: ${DUMP_ROOT} is ${NAS_FSTYPE} ${NAS_SOURCE}, write probe created + removed"
  fi
fi

# --- the farm: a real local directory (never a symlink out of the checkout) ---------- #
farm_ok() {   # <dir>: absent (DRY_RUN) or a real directory under the repo / harness
  [ ! -L "$1" ] || die "FARM GATE FAILED: ${1} is a symlink - abort"
  [ -e "$1" ] || return 0
  case "$(readlink -f -- "$1")" in "$(readlink -f "$REPO")"/?*|"${HREAL:-/nonexistent}"/?*) ;;
    *) die "FARM GATE FAILED: ${1} resolves outside this checkout - abort" ;; esac
}
farm_ok "$FARM_ROOT"

# --- per cell: checkpoint, link, (predicate), (run) ----------------------------------- #
declare -A SHA_DONE=()
verify_ckpt() {   # target expected-sha nas-manifest nas-key; hashed once per path
  local target="$1" want="$2" nasm="$3" key="$4" got
  [ -f "$target" ] || die "CHECKPOINT MISSING: ${target} - abort"
  [ -n "${SHA_DONE[$target]:-}" ] && return 0
  if [ -n "$nasm" ]; then
    grep -qxF "${want}  ${key}" "$nasm" 2>/dev/null || die "NAS MANIFEST CHECK FAILED: ${nasm} does not list '${want}  ${key}' - abort"
  fi
  got="$(sha256sum "$target" | awk '{print $1}')"
  [ "$got" = "$want" ] || die "CHECKPOINT SHA MISMATCH: ${target} is ${got}, the manifest pins ${want} - abort"
  SHA_DONE[$target]=1
  echo "checkpoint OK: ${target} (${want})"
}
cell_fields() {
  jq -r --arg c "$1" '.cells[] | select(.cell == $c) | [.arm, .block, .target, .sha256,
      (.nas_manifest // ""), (.nas_manifest_key // ""), .link, .metrics, .stream, (.canon // ""),
      .canon_probe, .receipt, (.dump | tostring), .dump_dir, .stem] | join("\u001f")' "$PLAN"
}
VERIFY_EXTRA=(); [ -n "$CANON_MANIFEST" ] && VERIFY_EXTRA=(--canon-ckpt-manifest "$CANON_MANIFEST")
existing_artifacts() {   # every artifact of the current cell that exists (one path per line)
  local p
  for p in "$METRICS" "$STREAM" "$CANON_PROBE" "$RECEIPT" "$DUMP_DIR" "$DUMP_DIR".staging-*; do
    case "$p" in *.staging-*.incomplete-*) continue ;; esac            # already parked
    { [ -e "$p" ] || [ -L "$p" ]; } && printf '%s\n' "$p"
  done
}
complete_and_verified() {   # 0 = SKIP; the collector runs is_complete exactly once
  local extra=("${VERIFY_EXTRA[@]}") p
  [ -f "$METRICS" ] && [ -f "$STREAM" ] && [ -f "$RECEIPT" ] || return 1
  [ -z "$CANON" ] || [ -f "$CANON" ] || return 1
  if [ "$DUMP" = true ]; then [ -d "$DUMP_DIR" ] && [ -f "$DUMP_DIR/COMPLETE" ] || return 1
  else extra+=(--no-dump); fi
  for p in "$DUMP_DIR".staging-*; do
    case "$p" in *.incomplete-*) ;; *) [ -e "$p" ] && return 1 ;; esac
  done
  "$PY" "$COLLECT" --verify-cell "$STEM" --pin "$EXPECT_HEAD" --cells "$CELLS" --farm-root "$FARM_ROOT" \
    --dump-root "$DUMP_ROOT" --base-dir "$REPO" --campaign-digest "$CAMPAIGN_DIGEST" "${extra[@]}" 2>&1 | grep -v -e pkg_resources -e declare_namespace | sed 's/^/    /'
  return "${PIPESTATUS[0]}"
}
prepare_cell() {   # sets the cell globals; 0 = run it, 3 = SKIP, 1 = refused (message printed)
  local cid="$1" what p dest
  IFS=$'\x1f' read -r ARM BLOCK TARGET SHA NASM NASK LINK METRICS STREAM CANON CANON_PROBE RECEIPT \
    DUMP DUMP_DIR STEM < <(cell_fields "$cid")
  mapfile -t ARGV < <(jq -r --arg c "$cid" '.cells[] | select(.cell == $c) | .argv[]' "$PLAN")
  ( verify_ckpt "$TARGET" "$SHA" "$NASM" "$NASK" ) || return 1          # die -> subshell exit 2
  SHA_DONE[$TARGET]=1
  if [ -L "$LINK" ] || [ -e "$LINK" ]; then
    [ -L "$LINK" ] && [ "$(readlink -- "$LINK")" = "$TARGET" ] \
      || { echo "FARM GATE FAILED: ${LINK} exists and is not a link to ${TARGET} - abort"; return 1; }
  elif [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$(dirname "$LINK")" && ( farm_ok "$(dirname "$LINK")" ) && ln -s -- "$TARGET" "$LINK" \
      || { echo "FARM GATE FAILED: cannot link ${LINK} - abort"; return 1; }
  fi
  if complete_and_verified; then echo "CELL ${cid} | ${BLOCK} | SKIP (complete and verified)"; return 3; fi
  mapfile -t STALE < <(existing_artifacts)
  what="run"; [ "${#STALE[@]}" -gt 0 ] && what="QUARANTINE ${#STALE[@]} artifact(s), then run"
  echo "CELL ${cid} | ${BLOCK} | ${ARM} | $([ "$DRY_RUN" = 1 ] && echo would) ${what}"
  for p in "${STALE[@]}"; do
    dest="${p}.incomplete-${TS}"
    if [ "$DRY_RUN" = 1 ]; then echo "  would quarantine ${p}"; continue; fi
    [ ! -e "$dest" ] && mv -- "$p" "$dest" || { echo "QUARANTINE FAILED: ${p} - abort"; return 1; }
    echo "  QUARANTINED ${p} -> ${dest}"
  done
  return 0
}
N_SEL="${#CELL_IDS[@]}"
if [ "$DRY_RUN" = 1 ]; then
  for cid in "${CELL_IDS[@]}"; do
    prepare_cell "$cid"; st=$?
    [ "$st" -eq 1 ] && exit 2
    [ "$st" -eq 0 ] && echo "DRYRUN ${cid} argv: $(jq -c --arg c "$cid" '.cells[] | select(.cell == $c) | .argv' "$PLAN")"
  done
  echo "=== DRY_RUN summary: ${N_SEL} cell(s) planned; nothing evaluated, linked, created or moved ==="
  exit 0
fi

run_cell() {   # lane cell sha spawn-digest argv... -- a background job: evaluate, then receipt + verify
  local lane="$1" cid="$2" sha="$3" spawn="$4"; shift 4
  local clog="$LOGDIR/depth_canon_${TS}_eval_${cid}.log" rc=0 post
  CUDA_VISIBLE_DEVICES="$lane" HF_HUB_OFFLINE=1 "$PY" "$EVAL_PROG" "$@" > "$clog" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then echo "  !! FAILED ${cid}: the evaluator exited rc=${rc} (log ${clog})"; return 1; fi
  post="$(tree_digest)" || post=""
  if [ "$post" != "$CAMPAIGN_DIGEST" ] || [ -n "$(tree_dirty)" ]; then
    echo "  !! FAILED ${cid}: the relevant tree changed while the cell ran (digest '${post}' != ${CAMPAIGN_DIGEST}); no receipt"
    return 4
  fi
  "$PY" "$COLLECT" --write-receipt "$cid" --plan-json "$PLAN" --cells "$CELLS" --base-dir "$REPO" \
    --lane "$lane" --log "$clog" --rc 0 --pin "$EXPECT_HEAD" --ckpt-sha "$sha" --farm-root "$FARM_ROOT" \
    --spawn-digest "$spawn" --post-digest "$post" \
    --dump-root "$DUMP_ROOT" "${VERIFY_EXTRA[@]}" --verify-after >> "$clog" 2>&1 \
    || { echo "  !! FAILED ${cid}: post-run verification refused the cell (log ${clog})"; return 1; }
  echo "  DONE ${cid} on GPU ${lane}: complete and verified"
}
declare -A LANE_PID=() PID_CELL=() PID_LANE=() GPU_NOTE=()
N_SKIP=0; N_DONE=0; FAILED=(); ABORT=""; DRIFT=""; NEXT=0; POLLS=0; i=0; READY=""
reap() {   # collect every finished job; never waits on a running one, never signals anything
  local pid st running=" $(jobs -rp | tr '\n' ' ') "
  for pid in "${!PID_CELL[@]}"; do
    case "$running" in *" ${pid} "*) continue ;; esac
    st=0; wait "$pid" || st=$?
    if [ "$st" -eq 0 ]; then N_DONE=$((N_DONE + 1)); else FAILED+=("${PID_CELL[$pid]}"); fi
    if [ "$st" -eq 4 ]; then DRIFT=1; ABORT="${ABORT:-PIN DRIFT: the relevant tree changed while ${PID_CELL[$pid]} ran}"; fi
    unset "LANE_PID[${PID_LANE[$pid]}]" "PID_CELL[$pid]" "PID_LANE[$pid]"
  done
}
RE_MAP='^([0-9]+),[[:space:]]*(GPU-[0-9A-Za-z-]+)$'
RE_APP='^([0-9]+),[[:space:]]*(GPU-[0-9A-Za-z-]+),[[:space:]]*([^,]*),[[:space:]]*([0-9]+|\[N/A\])$'
lane_state() {   # <gpu>: 0 usable | 1 not now (LANE_WHY) | 2 a query failed / is malformed (fail closed)
  local lane="$1" out rc line want pid uuid occ="" free
  local -A byidx=() byuuid=()
  out="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)"; rc=$?
  [ "$rc" -eq 0 ] && [ -n "$out" ] || { LANE_WHY="nvidia-smi --query-gpu=index,uuid failed (rc ${rc})"; return 2; }
  while IFS= read -r line; do
    [[ "$line" =~ $RE_MAP ]] || { LANE_WHY="malformed index,uuid row '${line}'"; return 2; }
    [ -z "${byidx[${BASH_REMATCH[1]}]:-}" ] && [ -z "${byuuid[${BASH_REMATCH[2]}]:-}" ] \
      || { LANE_WHY="duplicate index or uuid in the nvidia-smi map ('${line}')"; return 2; }
    byidx[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"; byuuid[${BASH_REMATCH[2]}]="${BASH_REMATCH[1]}"
  done <<<"$out"
  want="${byidx[$lane]:-}"
  [ -n "$want" ] || { LANE_WHY="GPU ${lane} has no uuid in the nvidia-smi index map"; return 2; }
  out="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; rc=$?
  [ "$rc" -eq 0 ] || { LANE_WHY="nvidia-smi --query-compute-apps failed (rc ${rc})"; return 2; }
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    [[ "$line" =~ $RE_APP ]] || { LANE_WHY="malformed compute-app row '${line}'"; return 2; }
    pid="${BASH_REMATCH[1]}"; uuid="${BASH_REMATCH[2]}"
    [ -n "${byuuid[$uuid]:-}" ] || { LANE_WHY="compute process ${pid} sits on an unknown uuid ${uuid}"; return 2; }
    [ "$uuid" = "$want" ] && occ+=" pid ${pid} (${BASH_REMATCH[3]}, ${BASH_REMATCH[4]} MiB, cwd $(readlink "/proc/${pid}/cwd" 2>/dev/null || echo '?'))"
  done <<<"$out"
  out="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$lane" 2>/dev/null)"; rc=$?
  [ "$rc" -eq 0 ] && [[ "$out" =~ ^[[:space:]]*([0-9]+)[[:space:]]*$ ]] \
    || { LANE_WHY="nvidia-smi memory.free of GPU ${lane} failed or is not one number (rc ${rc}: '${out//$'\n'/|}')"; return 2; }
  free="${BASH_REMATCH[1]}"
  [ "$free" -ge "$MIN_GPU_FREE_MIB" ] || { LANE_WHY="${free} MiB free < ${MIN_GPU_FREE_MIB} MiB"; return 1; }
  LANE_WHY=""
  if [ -n "$occ" ]; then
    [ "$ALLOW_SHARED_GPU" = 1 ] || { LANE_WHY="occupied by foreign compute process(es):${occ}"; return 1; }
    LANE_WHY="shared with${occ} (ALLOW_SHARED_GPU=1)"
  fi
  return 0
}
pick_lane() {   # LANE = the next idle lane (round-robin) lane_state accepts; a failed query aborts
  local k lane st; LANE=""
  for ((k = 0; k < ${#LANES[@]}; k++)); do
    lane="${LANES[$(( (NEXT + k) % ${#LANES[@]} ))]}"
    [ -z "${LANE_PID[$lane]:-}" ] || continue
    lane_state "$lane"; st=$?
    [ "$st" -eq 2 ] && { ABORT="GPU GATE FAILED: ${LANE_WHY}"; return 1; }
    if [ "$st" -eq 0 ]; then
      [ -z "$LANE_WHY" ] || echo "  GPU ${lane}: ${LANE_WHY}"
      LANE="$lane"; NEXT=$(( (NEXT + k + 1) % ${#LANES[@]} )); return 0
    fi
    if [ "${GPU_NOTE[$lane]:-}" != "${READY}#${POLLS}" ]; then   # once per waiting cell / poll
      echo "  GPU ${lane}: ${LANE_WHY} - waiting (nothing is ever signalled)"
      GPU_NOTE[$lane]="${READY}#${POLLS}"
    fi
  done
  return 1
}
gpu_wait() {   # one counted poll of the GPU gate; sets ABORT when the budget is spent
  POLLS=$((POLLS + 1))
  if [ "$GPU_MAX_POLLS" -gt 0 ] && [ "$POLLS" -ge "$GPU_MAX_POLLS" ]; then
    ABORT="GPU GATE: no usable lane (>= ${MIN_GPU_FREE_MIB} MiB free, unoccupied) after ${POLLS} poll(s)"; return
  fi
  sleep "$GPU_POLL_SECONDS"
}
while :; do
  reap
  if [ -n "$ABORT" ] || [ "$i" -ge "$N_SEL" ]; then [ "${#PID_CELL[@]}" -eq 0 ] && break; sleep 2; continue; fi
  if [ -z "$READY" ]; then
    prepare_cell "${CELL_IDS[$i]}"; st=$?
    [ "$st" -eq 1 ] && { ABORT="cell ${CELL_IDS[$i]} refused"; continue; }
    [ "$st" -eq 3 ] && { N_SKIP=$((N_SKIP + 1)); i=$((i + 1)); continue; }
    READY="${CELL_IDS[$i]}"
  fi
  [ "${#PID_CELL[@]}" -lt "${#LANES[@]}" ] || { sleep 2; continue; }
  if ! pick_lane; then
    [ -n "$ABORT" ] && continue
    if [ "${#PID_CELL[@]}" -gt 0 ]; then sleep 10; continue; fi   # a lane is busy: re-check soon
    gpu_wait; continue
  fi
  POLLS=0
  [ "$(git rev-parse HEAD)" = "$EXPECT_HEAD" ] || { ABORT="PIN DRIFT: HEAD moved during the campaign"; DRIFT=1; continue; }
  SPAWN_DIGEST="$(tree_digest)" || SPAWN_DIGEST=""     # the evaluated bytes, before EVERY spawn (Codex rB)
  if [ "$SPAWN_DIGEST" != "$CAMPAIGN_DIGEST" ] || [ -n "$(tree_dirty)" ]; then
    ABORT="PIN DRIFT: the relevant tree changed during the campaign (digest '${SPAWN_DIGEST}' != ${CAMPAIGN_DIGEST}); no further cell is spawned"
    DRIFT=1; continue
  fi
  lane_state "$LANE"; st=$?                  # re-checked immediately before the spawn (Codex rB HIGH)
  if [ "$st" -ne 0 ]; then
    [ "$st" -eq 2 ] && { ABORT="GPU GATE FAILED: ${LANE_WHY}"; continue; }
    echo "  GPU ${LANE}: changed at the re-check before the spawn (${LANE_WHY}) - not spawned"
    gpu_wait; continue
  fi
  echo "  RUN ${READY} on GPU ${LANE} (log $LOGDIR/depth_canon_${TS}_eval_${READY}.log)"
  run_cell "$LANE" "$READY" "$SHA" "$SPAWN_DIGEST" "${ARGV[@]}" &
  LANE_PID[$LANE]=$!; PID_CELL[$!]="$READY"; PID_LANE[$!]="$LANE"
  READY=""; i=$((i + 1))
done
COMPLETE=$((N_SKIP + N_DONE))
echo "=== summary: selected ${N_SEL} | skipped (complete and verified) ${N_SKIP} | ran ok ${N_DONE} | failed ${#FAILED[@]} | not run $((N_SEL - COMPLETE - ${#FAILED[@]})) ==="
for c in "${FAILED[@]}"; do echo "  FAILED ${c}"; done
echo "completion: ${COMPLETE}/${N_SEL} selected cell(s) complete and verified (manifest: $(jq '.cells | length' "$CELLS") cells)"
[ -z "$ABORT" ] || { echo "ABORTED: ${ABORT} - abort"; [ -z "$DRIFT" ] || exit 4; exit 2; }
[ "${#FAILED[@]}" -eq 0 ] && [ "$COMPLETE" -eq "$N_SEL" ] && { echo "CAMPAIGN SELECTION COMPLETE"; exit 0; }
echo "CAMPAIGN SELECTION INCOMPLETE - re-run to resume"
exit 1
