#!/bin/bash
# exp_14 ladder: the acceptance checks rungs 5, 6 and 7 share, in one place.
#
# Codex full-r2 finding 3. Round F gave every rung a `.done` marker it could not write
# unless it passed -- but the checks themselves were wrong about what a passing run looks
# like, and a wrong check is worse than none: rung 5's `max_steps=3 reached` matches
# NEITHER of the two recorded successful logs, because Lightning writes the line with
# backticks. Checks that gate GPU evidence must therefore be replayable against evidence
# that already exists, so they live here and are runnable on their own:
#
#   bash ladder_checks.sh fit-banner    <log> <steps>
#   bash ladder_checks.sh backbone      <log> <cyl|van>
#   bash ladder_checks.sh finite-loss   <log>
#   bash ladder_checks.sh batches       <log> <n/total>
#   bash ladder_checks.sh no-checkpoints <dir> [dir…]
#
# and are sourced by the rung scripts, which call the lc_* functions directly. Sourcing
# must not disturb the caller's shell options, so nothing here runs `set` unless this
# file is the executed script. Every check is read-only: it greps logs and lists files.
lc_say () { echo "ladder-check: $*" >&2; }

# Lightning's real line, verbatim:  `Trainer.fit` stopped: `max_steps=3` reached.
lc_fit_banner () {
  local log="${1:-}" steps="${2:-}"
  [ -f "$log" ] || { lc_say "no such log: $log"; return 1; }
  grep -qF '`Trainer.fit` stopped: `max_steps='"$steps"'` reached' "$log" && return 0
  lc_say "no Lightning '\`max_steps=$steps\` reached' banner in $log"; return 1; }

# WHICH geometry backbone actually loaded. The two arms differ only here, so a cyl config
# that silently fell back to the vanilla ViT would otherwise look like a clean cyl run.
lc_backbone () {
  local log="${1:-}" arm="${2:-}" want cyl='Loading cylindrical_dinov3 ViT from'
  [ -f "$log" ] || { lc_say "no such log: $log"; return 1; }
  case "$arm" in
    cyl) want="$cyl" ;;
    van) want='Loading ViT model from' ;;
    *) lc_say "unknown arm '$arm' (expected cyl or van)"; return 2 ;;
  esac
  grep -qF "$want" "$log" || { lc_say "no '$want' backbone banner in $log"; return 1; }
  if [ "$arm" = van ] && grep -qF "$cyl" "$log"; then
    lc_say "the van log $log loaded the CYLINDRICAL backbone"; return 1; fi
  return 0; }

# A run that reached its steps but produced nan/inf learned nothing; the progress bar
# carries the last value, so the last `train/loss=` in the log is the one that counts.
lc_finite_loss () {
  local log="${1:-}" value
  [ -f "$log" ] || { lc_say "no such log: $log"; return 1; }
  value=$( { grep -o 'train/loss=[^], ]*' "$log" || true; } | tail -1 | sed 's/^train.loss=//')
  [ -n "$value" ] || { lc_say "no train/loss at all in $log"; return 1; }
  case "$value" in                    # nan / inf carry letters the numeric set excludes
    ''|*[!0-9.eE+-]*) lc_say "train/loss=$value is not a finite number in $log"; return 1 ;;
  esac
  return 0; }

# The batch the run actually reached, e.g. 5/1148 -- an exit code alone proves nothing
# about how far the fit got, and `15/1148` must not satisfy `5/1148`.
lc_batches () {
  local log="${1:-}" want="${2:-}"
  [ -f "$log" ] || { lc_say "no such log: $log"; return 1; }
  grep -qE "(^|[^0-9/])${want}([^0-9]|\$)" "$log" && return 0
  lc_say "the batch count '$want' never appears in $log"; return 1; }

lc_no_checkpoints () {
  local n
  n=$( { find "$@" -name '*.ckpt' 2>/dev/null || true; } | wc -l)
  [ "$n" = 0 ] && return 0
  lc_say "$n checkpoint(s) under $* from a rung that must write none"; return 1; }

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -uo pipefail
  CMD="${1:-}"; [ "$#" -gt 0 ] && shift
  case "$CMD" in
    fit-banner)     lc_fit_banner "$@" ;;
    backbone)       lc_backbone "$@" ;;
    finite-loss)    lc_finite_loss "$@" ;;
    batches)        lc_batches "$@" ;;
    no-checkpoints) lc_no_checkpoints "$@" ;;
    *) lc_say "usage: ladder_checks.sh {fit-banner|backbone|finite-loss|batches|no-checkpoints} …"
       exit 2 ;;
  esac
fi
