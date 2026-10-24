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
#   run         (cycle 4c) one lane per GPU, gated on free memory; never kills anything.
#
# Usage:  EXPECT_HEAD=<40-hex pin> bash depth_canon_eval_run.sh [--blocks B,..]
#           [--only CELL,..] [--limit N] [--no-dump-blocks B,..] [--dump-root DIR]
#           [--lanes 0,1] [--canon-ckpt-manifest PATH]
#   DRY_RUN=1  every gate except the side-effecting / GPU ones: no link, no mkdir, no
#              quarantine, no evaluation; prints each cell's exact argv.
#   GPU_POLL_SECONDS (default 600), GPU_MAX_POLLS (default 0 = wait for ever).
#   DEPTH_CANON_EVAL_HARNESS=<dir under /tmp holding a .depth_canon_eval_harness marker>:
#              test mode only -- DEPTH_CANON_EVAL_PY (stub evaluator) and DEPTH_CANON_CELLS
#              are honoured, and the farm / logs / lock move into that dir.
# Exit: 0 all selected cells complete (or DRY_RUN) | 1 a cell failed | 2 a gate refused.
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
DIRTY="$(git status --porcelain --untracked-files=all -- eval_FLAC.py src ':(exclude)src/tests' data/AR \
  "${MODEL_CFGS[@]}" "$SELF" "$COLLECT" "$MAKE_CELLS" "$EXPDIR/depth_canon_cells.json")"
[ -z "$DIRTY" ] || { echo "EVALUATOR TREE NOT COMMITTED at HEAD (every cell must run the pinned code):"; echo "$DIRTY"; exit 2; }
echo "evaluator pin OK: ${HEAD_SHA}"
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
PLAN_ARGS=(--plan "$PLAN" --cells "$CELLS" --farm-root "$FARM_ROOT" --no-dump-blocks "$NO_DUMP")
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
[ "$DRY_RUN" = 1 ] || die "RUN PATH NOT YET AVAILABLE in this revision (DRY_RUN=1 only) - abort"
N_SEL="${#CELL_IDS[@]}"
for cid in "${CELL_IDS[@]}"; do
  IFS=$'\x1f' read -r ARM BLOCK TARGET SHA NASM NASK LINK METRICS STREAM CANON CANON_PROBE RECEIPT \
    DUMP DUMP_DIR STEM < <(cell_fields "$cid")
  verify_ckpt "$TARGET" "$SHA" "$NASM" "$NASK"
  if [ -L "$LINK" ] && [ "$(readlink -- "$LINK")" != "$TARGET" ]; then
    die "FARM GATE FAILED: ${LINK} -> $(readlink -- "$LINK"), not ${TARGET} - abort"
  fi
  echo "CELL ${cid} | ${BLOCK} | ${ARM} | would run | link ${LINK} -> ${TARGET}"
  echo "DRYRUN ${cid} argv: $(jq -c --arg c "$cid" '.cells[] | select(.cell == $c) | .argv' "$PLAN")"
done
echo "=== DRY_RUN summary: ${N_SEL} cell(s) planned; nothing evaluated, linked, created or moved ==="
exit 0
