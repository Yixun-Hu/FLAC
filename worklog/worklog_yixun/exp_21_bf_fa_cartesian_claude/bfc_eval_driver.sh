#!/usr/bin/env bash
# ============================================================================
# bfc_eval_driver.sh — exp_21's whole evaluation campaign, driven from ONE
# protocol definition.
#
# 34 cells, in this order:
#   10  BFC registered      5 seeds x {K1, K8}      -> the Table row pair
#    4  BFC invariance grid K8/s42 at 45/90/180/270 -> plan §5 acceptance test
#   10  BFre (B-F @40k)     5 seeds x {K1, K8}      -\  D6: both comparators
#   10  P1re (P1  @40k)     5 seeds x {K1, K8}      -/  re-evaluated at THIS pin
#
# The grid's 0-degree member is the registered K8/seed-42 cell itself — it
# carries --record-stream like every registered cell now does, so re-running it
# under a second name would publish two measurements of one thing (plan §5:
# "14 unique BFC cells — 10 registered + 4 extra grid angles, the K8/s42/0 cell
# shared"). Hence 34 invocations, not 35.
#
# THIS SCRIPT RESTATES NO FLAG. Every command comes from exp21_protocol.py, which
# imports its constants from exp21_validate_cell.py — the same module the model-
# comparison table admits rows with. That is the whole design: the thing that
# RUNS the evaluation and the thing that JUDGES it cannot disagree, because there
# is one definition (announcement 05: eval-protocol flags are part of the
# experiment, never defaults; `--cond-autocast bf16` alone moved the same
# checkpoint's T60 between 8.202 and 10.652 in the exp_10 record).
#
# Modes (env):
#   DRY_RUN=1   print all 34 commands and exit. Touches nothing, needs no GPU.
#               PLACEHOLDER=<s> renders BFC's unresolved epoch number as <s>, so
#               the inventory can be reviewed before the arm has trained.
#   ARM=<name>  restrict to one of BFC | BFre | P1re.
#
# Resume is by ARTIFACT ADMISSION, not by a bookmark: a cell is skipped only when
# its metrics JSON already exists AND validates against the registered protocol
# (exp_17's roteval runners resume the same way). A half-written or protocol-wrong
# artifact is therefore re-run rather than trusted, and re-running the driver
# after any interruption is safe.
#
# Written by the Coder seat (Claude Opus 5, max effort), exp_21 round 5.
# ============================================================================
set -uo pipefail

EXPDIR="worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude"
PROTOCOL="${EXPDIR}/exp21_protocol.py"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG="${EXPDIR}/bf_fa_cartesian_${TS}_eval_driver.log"

DRY_RUN="${DRY_RUN:-0}"
ARM="${ARM:-}"
PLACEHOLDER="${PLACEHOLDER:-}"

_bool01() { case "$2" in 0|1) return 0;; *) echo "$1 must be 0 or 1, got '$2' - abort"; return 1;; esac; }
_bool01 DRY_RUN "$DRY_RUN" || exit 2
case "$ARM" in ""|BFC|BFre|P1re) ;; *) echo "ARM must be BFC, BFre or P1re, got '$ARM' - abort"; exit 2;; esac

[ -f "$PROTOCOL" ] || { echo "missing ${PROTOCOL} - abort"; exit 2; }
[ -f "eval_FLAC.py" ] || { echo "run from the repo root (no ./eval_FLAC.py here) - abort"; exit 2; }

ARM_ARGS=(); [ -n "$ARM" ] && ARM_ARGS=(--arm "$ARM")
PH_ARGS=();  [ -n "$PLACEHOLDER" ] && PH_ARGS=(--placeholder "$PLACEHOLDER")

# ---------------------------------------------------------------------------
# DRY RUN: the inventory, verbatim, nothing else.
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = "1" ]; then
  python "$PROTOCOL" commands "${ARM_ARGS[@]}" "${PH_ARGS[@]}" || exit 2
  exit 0
fi

# ---------------------------------------------------------------------------
# Real run.
# ---------------------------------------------------------------------------
exec > >(tee -a "$LOG") 2>&1
echo "=== exp_21 eval driver started $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "protocol: ${PROTOCOL} | arm filter: ${ARM:-<all>} | log: ${LOG}"
echo "source_sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Refuse to share the GPUs with another evaluation: two eval_FLAC runs at batch
# 64 do not fit, and a half-run cell is worse than an unrun one.
if pgrep -af "python .*eval_FLAC\.py" | grep -v "$$" >/dev/null 2>&1; then
  echo "REFUSING: another eval_FLAC.py is already running:"
  pgrep -af "python .*eval_FLAC\.py"
  exit 2
fi

# --- per-arm trained-as preflight (r5 full review, findings 1 + 4) ----------
# One CPU checkpoint read per arm, BEFORE any of its cells: a wrong-arm campaign
# then costs one read instead of ten evaluations. eval_FLAC enforces the BFC
# contract itself on every cell; this is what holds the two COMPARATOR
# checkpoints to their own (different, historically-shaped) contracts.
for arm in BFC BFre P1re; do
  [ -n "$ARM" ] && [ "$ARM" != "$arm" ] && continue
  echo "--- trained-as preflight: ${arm} ---"
  python "$PROTOCOL" preflight "$arm" || {
    echo "preflight FAILED for ${arm} - refusing to evaluate this arm"; exit 2; }
done

# --- the cells ---------------------------------------------------------------
mapfile -t CELLS < <(python "$PROTOCOL" commands "${ARM_ARGS[@]}" --with-paths) || exit 2
[ "${#CELLS[@]}" -gt 0 ] || { echo "no cells to run - abort"; exit 2; }
echo "cells to consider: ${#CELLS[@]}"

ran=0; skipped=0; failed=0
for entry in "${CELLS[@]}"; do
  metrics_json="${entry%%$'\t'*}"
  cmd="${entry#*$'\t'}"
  name="$(basename "$metrics_json")"

  # ADMISSION-GATE RESUME: an existing artifact is trusted only if it still
  # validates as this cell's registered measurement AND carries its full-split
  # sidecar. Anything else is re-run.
  if [ -f "$metrics_json" ]; then
    if python - "$metrics_json" <<'PY'
import importlib.util, json, os, sys
here = os.path.join("worklog", "worklog_yixun", "exp_21_bf_fa_cartesian_claude")
spec = importlib.util.spec_from_file_location(
    "exp21_validate_cell", os.path.join(here, "exp21_validate_cell.py"))
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
path = sys.argv[1]
try:
    record = json.load(open(path))
except Exception as exc:
    print(f"  existing artifact unreadable ({exc})"); sys.exit(1)
reasons = V.stream_sidecar_reasons(path)
name = str(record.get("eval_name") or "")
try:
    reasons += V.validate_metrics_record(record, V.parse_eval_name(name))
except ValueError:
    # a comparator or grid cell: not a registered TABLE cell, so the row
    # validator cannot judge it. The sidecar + a finite metrics block is what
    # can be checked here; the table gate judges the rows themselves.
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        reasons.append("no metrics block")
    if record.get(V.CKPT_SHA_FIELD) is None:
        reasons.append(f"no {V.CKPT_SHA_FIELD}")
    if record.get(V.TRAINED_FIELD) is None:
        reasons.append(f"no {V.TRAINED_FIELD}")
for r in reasons:
    print("  " + r)
sys.exit(1 if reasons else 0)
PY
    then
      echo "SKIP (already valid): ${name}"; skipped=$((skipped + 1)); continue
    else
      echo "RE-RUN (existing artifact did not validate): ${name}"
    fi
  fi

  echo "--- $(date '+%H:%M:%S') RUN: ${name}"
  echo "    ${cmd}"
  # Split into an argv array rather than running an unquoted ${cmd}: word
  # splitting alone is what is wanted here, and bare expansion would ALSO
  # pathname-expand any token containing a glob character.
  IFS=' ' read -r -a _tok <<< "$cmd"
  if "${_tok[@]}"; then ran=$((ran + 1)); else
    rc=$?; failed=$((failed + 1)); echo "    CELL FAILED (rc=${rc}): ${name}"
  fi
done

echo "=== exp_21 eval driver done $(date '+%Y-%m-%d %H:%M:%S'): ${ran} run, ${skipped} skipped, ${failed} failed ==="
[ "$failed" -eq 0 ] || exit 1
exit 0
