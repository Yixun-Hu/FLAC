#!/usr/bin/env bash
# ============================================================================
# fa_orbit_chunk_watchdog.sh — login-side driver for exp_11's CHUNKED legs.
#
# WHY. The partition never backfills a 34-160 h allocation, so the 40k -> 100k
# extension legs sat PD indefinitely. A chunked leg asks for hours instead: it
# trains to the next 2500-step boundary, saves, and exits. This script is what
# turns a sequence of such jobs into one continuous run — it watches for an arm
# with no live job, records the chunk that just finished, and submits the next.
#
# WHAT IT MAY DO. Exactly three things: read (squeue/sacct/ls/the registry),
# append to its own log and state file, and call the two sanctioned tools —
# fa_orbit_record_restart.py and fa_orbit_submit.sh. It NEVER calls sbatch
# itself, never deletes or rewrites a checkpoint, a manifest or the registry
# (the recorder owns the registry), and never edits the tracked tree. It also
# never runs fa_orbit_add_anchor.py: anchoring an arm is an OPERATOR action
# (it audits a 40k checkpoint into the lineage), so an arm with no anchor is
# frozen at startup instead (round-5 review B7).
#
# FAIL-CLOSED. Every step is refused rather than guessed: an ambiguous manifest
# or checkpoint, a recorder refusal, or a submitter refusal counts as a failure
# for that arm, and MAX_CONSEC_FAIL consecutive failures FREEZE the arm until a
# human removes its frozen_<ARM> line from the state file. A frozen arm is
# skipped loudly, so the log says why nothing is progressing. The chunk chain
# itself is guarded independently of this script: fa_orbit_ckpt_preflight.py
# --chain refuses a chunk whose predecessor was never recorded, so a watchdog
# bug cannot launder an unrecorded checkpoint into the lineage.
#
# ROUND-5 REVIEW FIXES.
#   B2 singleton + no TOCTOU: one instance per experiment folder (flock on
#      .chunk_watchdog.lock, held for the whole run on fd 8), the arm list is
#      de-duplicated, a FAILING squeue is UNKNOWN (skip) rather than "empty",
#      and the last submitted job is re-checked BY ID before a new submission.
#   B3 settlement: lastjob_<ARM> is retained until sacct reports a TERMINAL
#      state; an empty/lagging/nonterminal sacct skips the arm this poll. The
#      finished chunk's manifest is located BY THE SETTLED JOB ID, so a failed
#      attempt and its retry can never be confused. CANCELLED is a human
#      intervention signal: the arm freezes immediately, it is never retried.
#   B4 terminal chunk: the record reconciliation runs BEFORE the DONE test, so
#      the last chunk (97500 -> 100000) is recorded like every other one. DONE
#      requires the newest checkpoint to be exactly TARGET *and* the recorded
#      chain tip to be TARGET; a checkpoint past TARGET freezes the arm.
#   B7 anchors: every selected arm's audited anchor (final_ckpt_sha256 +
#      final_step) is validated at STARTUP, before anything is submitted.
#
# ROUND-5 r2 REVIEW FIXES.
#   B1 the anti-duplicate reservation MOVED INTO fa_orbit_submit.sh, where it
#      also covers manual invocations: the submitter takes an exclusive flock on
#      .submit_<ARM>.lock, re-checks the queue INSIDE it, and only then sbatches.
#      The submitter is therefore the authority on duplicates; this script's own
#      name-scoped squeue is a FAST-PATH SKIP, nothing more (the unreachable
#      by-ID recheck it used to carry has been removed). lastjob_<ARM> is
#      persisted IMMEDIATELY after a successful submit, not at end-of-pass, so a
#      crash in the same poll cannot lose the job we just queued.
#   B3 versioned checkpoints: Lightning writes `epoch=E-step=N-v1.ckpt` when the
#      unversioned name is taken, so BOTH name shapes are parsed for the newest
#      step, and the next chunk resumes the recorded tip's final_ckpt_path (the
#      unique on-disk file is used only for the 40k anchor, which has no link).
#   B7+ the startup anchor check is DEEP: final_step == 40000 exactly, a 64-hex
#      lowercase sha, the file present in the registry's canonical directory, and
#      the file HASHED ONCE at startup and compared. Five ~700 MB hashes cost
#      about a minute; a mismatch freezes that arm.
#   Per-arm MAX chunk (C4L/C8/C16/C32 2500, VANL 5000): a chunk longer than the
#      arm's PINNED_TIME_LIMIT_CHUNK_<ARM> was sized for cannot finish inside its
#      allocation, so it is rejected with the pin named.
#
# LOCK LIFETIME CAVEAT (round-5 r2 non-blocking). The singleton lock is held on
# fd 8, which CHILD PROCESSES INHERIT — including the `sleep` between polls. If
# the watchdog is killed while a child still runs, the kernel keeps the lock
# until that child also exits, so "released when the watchdog dies" is really
# "released when the watchdog and its current child are both gone" (at most one
# POLL interval). This is safe in the direction that matters — a second watchdog
# still cannot start while the first is alive — but the release can be delayed.
#
# USAGE (KEY=VALUE arguments only; every key whitelisted, no value is ever eval'd)
#   bash fa_orbit_chunk_watchdog.sh                       # the pinned defaults
#   bash fa_orbit_chunk_watchdog.sh ARMS=C4L,C8 POLL=600
#   bash fa_orbit_chunk_watchdog.sh ONESHOT=1 DRYRUN=1    # one pass, submits nothing
#
#   ARMS=C4L,C8,C16,C32,VANL  CHUNK=2500  TARGET=100000  POLL=300
#   MAX_CONSEC_FAIL=2  ONESHOT=0  DRYRUN=0
#   PER-ARM CHUNK: CHUNK_C4L= CHUNK_C8= CHUNK_C16= CHUNK_C32= CHUNK_VANL=5000
#     A leg pays 10-15 min of startup whatever its length, so a 2500-step chunk
#     is 20-28% overhead for the fast arm (VANL) and ~4% for the slow one (C32).
#     VANL therefore DEFAULTS to 5000 (it still fits its 02:30:00 chunk pin);
#     every other arm defaults to the global CHUNK. An explicit CHUNK= sets the
#     default for the other four arms only — to change VANL, pass CHUNK_VANL=.
#     The submitted chunk end is capped at TARGET either way.
#     MAXIMUM per arm (time-pin compatibility, refused above it):
#       C4L 2500  C8 2500  C16 2500  C32 2500  VANL 5000
#   TEST HOOKS (change no decision, only where the script reads/writes):
#   OUTPUT_ROOT=outputs_FLAC  REGISTRY=<expdir>/arm_launch_registry.json
#   STATE=<expdir>/.chunk_watchdog_state  LOG=<expdir>/fa_orbit_chunk_watchdog.log
#
# COST. One `squeue` and one `ls` per arm per poll (plus, once an arm is above
# 40000, one short python read of the registry). Nothing else runs on the login
# node; the training itself is entirely inside Slurm.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SUBMITTER="${EXPDIR}/fa_orbit_submit.sh"
RECORDER="${EXPDIR}/fa_orbit_record_restart.py"
PY=/n/fs/gatrdp/envs/flac/bin/python
ANCHOR_STEP=40000                 # where every arm's INITIAL run ended

ARMS="C4L,C8,C16,C32,VANL"; CHUNK=2500; TARGET=100000; POLL=300
MAX_CONSEC_FAIL=2; ONESHOT=0; DRYRUN=0
# Per-arm chunk size. Empty = "use the global CHUNK"; VANL is 5000 by default
# (its 2500-step leg would be ~a quarter startup overhead — round-5 NON-BLOCKING).
CHUNK_C4L=""; CHUNK_C8=""; CHUNK_C16=""; CHUNK_C32=""; CHUNK_VANL=5000
OUTPUT_ROOT="outputs_FLAC"
REGISTRY="${EXPDIR}/arm_launch_registry.json"
STATE="${EXPDIR}/.chunk_watchdog_state"
LOG="${EXPDIR}/fa_orbit_chunk_watchdog.log"
# B2: the singleton lock is FIXED to the experiment folder — the resource being
# protected is "the arms of exp_11", not any particular state file, so it must
# not be relocatable by an argument.
LOCKFILE="${EXPDIR}/.chunk_watchdog.lock"

# --- argument parsing: whitelist the KEY, shape-check the VALUE, never eval ---
reject()   { echo "$1" >&2; exit 2; }
is_num()   { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
is_armset() {
  local v="$1" a
  [ -n "$v" ] || return 1
  local IFS=,
  for a in $v; do
    case "$a" in C4L|C8|C16|C32|VANL) ;; *) return 1 ;; esac
  done
}
for kv in "$@"; do
  case "$kv" in *=*) ;; *) reject "argument '${kv}' is not KEY=VALUE" ;; esac
  key="${kv%%=*}"; val="${kv#*=}"
  case "$key" in
    ARMS)            is_armset "$val" || reject "ARMS='${val}' is not a comma-separated list of C4L|C8|C16|C32|VANL" ;;
    CHUNK|TARGET|POLL|MAX_CONSEC_FAIL|CHUNK_C4L|CHUNK_C8|CHUNK_C16|CHUNK_C32|CHUNK_VANL)
                     is_num "$val" || reject "${key}='${val}' is not a non-negative integer" ;;
    ONESHOT|DRYRUN)  case "$val" in 0|1) ;; *) reject "${key}='${val}' must be 0 or 1" ;; esac ;;
    OUTPUT_ROOT|REGISTRY|STATE|LOG)
                     case "$val" in
                       ''|*[!A-Za-z0-9/._-]*) reject "${key}='${val}' has unsafe characters" ;;
                     esac ;;
    *)               reject "unknown argument '${kv}' (expected ARMS=/CHUNK=/CHUNK_<ARM>=/TARGET=/POLL=/MAX_CONSEC_FAIL=/ONESHOT=/DRYRUN=/OUTPUT_ROOT=/REGISTRY=/STATE=/LOG=)" ;;
  esac
  printf -v "$key" '%s' "$val"      # name whitelisted above; value never parsed
done
[ "$CHUNK" -gt 0 ] || reject "CHUNK must be positive"
[ "$((CHUNK % 2500))" -eq 0 ] || reject "CHUNK=${CHUNK} is not a multiple of the 2500-step checkpoint cadence"
# every per-arm override gets the SAME shape checks as the global one
for A in C4L C8 C16 C32 VANL; do
  eval "PERARM=\${CHUNK_${A}}"                       # name from a literal whitelist
  [ -n "$PERARM" ] || continue
  [ "$PERARM" -gt 0 ] || reject "CHUNK_${A} must be positive"
  [ "$((PERARM % 2500))" -eq 0 ] || reject "CHUNK_${A}=${PERARM} is not a multiple of the 2500-step checkpoint cadence"
done
[ "$TARGET" -gt "$ANCHOR_STEP" ] || reject "TARGET=${TARGET} must exceed the ${ANCHOR_STEP} anchor"
[ "$((TARGET % 2500))" -eq 0 ] || reject "TARGET=${TARGET} is not a multiple of 2500"
[ "$MAX_CONSEC_FAIL" -ge 1 ] || reject "MAX_CONSEC_FAIL must be at least 1"
[ -f "$SUBMITTER" ] || reject "missing ${SUBMITTER}"
[ -f "$RECORDER" ] || reject "missing ${RECORDER}"
[ -f "$REGISTRY" ] || reject "missing ${REGISTRY}"

# --- B2(c): the arm list is DE-DUPLICATED ------------------------------------
# ARMS=C8,C8 is two passes over one arm in one poll: the first submits, the
# second sees the job it just queued only if the scheduler is already listing it.
# Built HERE (before the singleton lock) so the chunk-size checks below can be
# expressed per SELECTED arm and still refuse before touching any shared state.
IFS=',' read -r -a ARM_RAW <<< "$ARMS"
ARM_LIST=()
for A in "${ARM_RAW[@]}"; do
  SEEN=0
  for B in ${ARM_LIST[@]+"${ARM_LIST[@]}"}; do [ "$A" = "$B" ] && SEEN=1; done
  [ "$SEEN" -eq 0 ] && ARM_LIST+=("$A")
done
[ "${#ARM_LIST[@]}" -gt 0 ] || reject "ARMS='${ARMS}' selected no arm"

# --- per-arm MAXIMUM chunk: the wall pin is what makes a chunk feasible -------
# NON-BLOCKING adoption, round-5 r2. A chunk leg is walled by
# PINNED_TIME_LIMIT_CHUNK_<ARM>, and each of those pins was sized for a specific
# chunk length: 2500 steps for the four orbit arms, 5000 for the (much faster)
# vanilla arm. A longer chunk than its pin was sized for cannot reach its
# boundary inside the allocation — the leg is wall-killed, no boundary
# checkpoint is written, and the chain stalls at that step forever. So the table
# below is a compatibility constraint, not a preference, and the refusal names
# the pin the request would have violated.
max_chunk_for() { case "$1" in VANL) printf '5000' ;; *) printf '2500' ;; esac; }
for A in "${ARM_LIST[@]}"; do
  eval "EFF=\${CHUNK_${A}:-}"            # name from the is_armset whitelist
  SRC="CHUNK_${A}"
  [ -n "$EFF" ] || { EFF="$CHUNK"; SRC="CHUNK"; }
  MAXC="$(max_chunk_for "$A")"
  [ "$EFF" -le "$MAXC" ] || reject "${SRC}=${EFF} gives ${A} a ${EFF}-step chunk, above the ${MAXC}-step maximum its PINNED_TIME_LIMIT_CHUNK_${A} wall pin was sized for — a longer chunk cannot reach its boundary inside the allocation the submitter requests, and a wall-killed leg writes no boundary checkpoint"
done

# --- B2(a): ONE watchdog per experiment folder --------------------------------
# Two watchdogs are a double-submission engine: both see "no live job", both
# submit the same boundary, and the job's own run-directory flock cannot help
# because it is taken long after scheduling. The lock is held on fd 8 for this
# process's whole lifetime and released by the kernel when it exits, so a killed
# watchdog leaves nothing stale behind.
exec 8>"$LOCKFILE" || reject "could not open the watchdog lock ${LOCKFILE}"
flock -n 8 || reject "another chunk watchdog already holds ${LOCKFILE} — refusing to start a second instance (two watchdogs would submit the same chunk twice)"

log() { printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG"; echo "$1"; }

chunk_for() {   # <arm> -> that arm's chunk size (per-arm override, else global)
  local v; eval "v=\${CHUNK_$1:-}"         # $1 is whitelisted by is_armset
  [ -n "$v" ] && { printf '%s' "$v"; return 0; }
  printf '%s' "$CHUNK"
}

# --- state: a plain KV file a human can read and edit ------------------------
# keys: fail_<ARM> <n> | frozen_<ARM> <when>|<reason> | lastjob_<ARM> <jid> |
#       done_<ARM> <step>@<target>.  Removing a frozen_<ARM> line un-freezes that
#       arm AND clears its failure streak (see load_state). done_<ARM> carries the
#       TARGET it was reached against, so reusing a state file with a larger
#       TARGET does not read the old completion as this campaign's.
declare -A ST=()
load_state() {
  ST=()
  [ -f "$STATE" ] || return 0
  local k v a
  while read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    ST["$k"]="$v"
  done < "$STATE"
  # NON-BLOCKING adoption: a human who deletes a frozen_<ARM> line means "try
  # this arm again", but the fail counter that caused the freeze survived and
  # the very next failure re-froze the arm immediately. Clearing a freeze now
  # clears the streak too. Deliberately NARROWER than "reset whenever frozen is
  # absent": a sub-threshold streak (fail < MAX_CONSEC_FAIL, never frozen) must
  # survive across polls or MAX_CONSEC_FAIL could never be reached at all.
  for a in C4L C8 C16 C32 VANL; do
    if [ -z "${ST[frozen_$a]:-}" ] && [ "${ST[fail_$a]:-0}" -ge "$MAX_CONSEC_FAIL" ] 2>/dev/null; then
      ST["fail_$a"]="0"
      log "${a}: frozen_${a} was cleared by hand — its ${MAX_CONSEC_FAIL}-failure streak is reset too"
    fi
  done
}
save_state() {
  local tmp="${STATE}.tmp.$$" k
  {
    echo "# exp_11 chunk watchdog state (plain KV, rewritten atomically)."
    echo "# Delete a frozen_<ARM> line to let the watchdog submit that arm again;"
    echo "# doing so also resets that arm's fail_<ARM> streak to 0 on the next poll."
    if [ "${#ST[@]}" -gt 0 ]; then
      for k in "${!ST[@]}"; do printf '%s %s\n' "$k" "${ST[$k]}"; done | sort
    fi
  } > "$tmp" || { echo "could not write ${tmp}" >&2; return 1; }
  mv -f "$tmp" "$STATE" || { echo "could not publish ${STATE}" >&2; return 1; }
}

bump_fail() {   # <arm> <reason>
  local arm="$1" reason="$2" n
  n=$(( ${ST[fail_$arm]:-0} + 1 ))
  ST["fail_$arm"]="$n"
  log "${arm}: FAILURE (${reason}) — ${n}/${MAX_CONSEC_FAIL} consecutive"
  if [ "$n" -ge "$MAX_CONSEC_FAIL" ]; then
    ST["frozen_$arm"]="$(date -Is)|${reason}"
    log "${arm}: !!! FROZEN after ${n} consecutive failures (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
  fi
}

freeze_now() {  # <arm> <reason> — a freeze that is NOT a retryable failure
  local arm="$1" reason="$2"
  ST["frozen_$arm"]="$(date -Is)|${reason}"
  log "${arm}: !!! FROZEN immediately (${reason}). No further submission for this arm until its frozen_${arm} line is removed from ${STATE}."
}

clear_fail() {  # <arm> <why>
  if [ "${ST[fail_$1]:-0}" != "0" ]; then
    log "${1}: failure streak cleared (${2})"
  fi
  ST["fail_$1"]="0"
}

# B3 (round-5 r2): BOTH Lightning name shapes count. A retry at a boundary whose
# unversioned name already exists saves `epoch=E-step=N-v1.ckpt`, and a parser
# that ignored those would read the chain as stuck one boundary back.
newest_ckpt_step() {   # <ckpt dir> -> the largest step with a checkpoint file
  ls -1 "$1" 2>/dev/null \
    | sed -n -e 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)\.ckpt$/\1/p' \
             -e 's/^epoch=[0-9]\{1,\}-step=\([0-9]\{1,\}\)-v[0-9]\{1,\}\.ckpt$/\1/p' \
    | sort -n | tail -1
}

ckpt_files_at() {      # <ckpt dir> <step> -> every checkpoint file at that step
  local d="$1" s="$2" f
  for f in "$d"/*-step="${s}".ckpt "$d"/*-step="${s}"-v*.ckpt; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done
  return 0
}

chain_tip_step() {     # <arm> -> the arm's recorded chain tip (or its anchor, or 0)
  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
import json, sys
row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
chain = row.get("chain") or []
print(int((chain[-1].get("final_step") if chain else row.get("final_step")) or 0))
PY
}

chain_tip_path() {     # <arm> -> the recorded tip's final_ckpt_path ('' if no chain)
  "$PY" - "$REGISTRY" "$1" <<'PY' 2>/dev/null
import json, sys
row = (json.load(open(sys.argv[1])).get("arms") or {}).get(sys.argv[2]) or {}
chain = row.get("chain") or []
print((chain[-1].get("final_ckpt_path") if chain else "") or "")
PY
}

# B7+ (round-5 r2): the startup anchor check is DEEP, not a presence test. It
# proves the arm's audited anchor is a real, unambiguous 40k checkpoint sitting
# where the recorder and preflight will look for it, and that its bytes still
# hash to the audited value. The canonical directory is derived from the
# REGISTRY's own save_dir (what the recorder and preflight use), which in
# production is exactly OUTPUT_ROOT/exp11_<ARM>/... — the same directory this
# script polls. One ~700 MB hash per arm, once, at startup.
anchor_state() {       # <arm> -> OK | <a one-line reason> | UNREADABLE
  "$PY" - "$REGISTRY" "$1" "$EXPDIR" <<'PY' 2>/dev/null || echo UNREADABLE
import hashlib, json, os, re, sys
reg_path, arm, expdir = sys.argv[1:4]
sys.path.insert(0, expdir)
from fa_orbit_ckpt_preflight import canonical_ckpt_dir      # noqa: E402
row = (json.load(open(reg_path)).get("arms") or {}).get(arm) or {}
sha, step = row.get("final_ckpt_sha256"), row.get("final_step")
if not sha or step is None:
    print("MISSING"); raise SystemExit(0)
if str(step) != "40000":
    print(f"the audited anchor is step {step!r}, not the 40000 every INITIAL run ended at")
    raise SystemExit(0)
if not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
    print(f"the audited final_ckpt_sha256 {str(sha)[:20]!r} is not a 64-char lowercase hex digest")
    raise SystemExit(0)
ckdir = canonical_ckpt_dir(row.get("save_dir", ""), arm, os.getcwd())
hits = [f for f in sorted(os.listdir(ckdir)) if re.search(r"-step=40000(-v\d+)?\.ckpt$", f)] \
    if os.path.isdir(ckdir) else []
if len(hits) != 1:
    print(f"expected exactly one step=40000 checkpoint in {ckdir}, found {len(hits)}"
          + (": " + ", ".join(hits) if hits else ""))
    raise SystemExit(0)
h = hashlib.sha256()
with open(os.path.join(ckdir, hits[0]), "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 22), b""):
        h.update(blk)
got = h.hexdigest()
print("OK" if got == sha
      else f"{hits[0]} hashes {got[:12]}, not the audited anchor {str(sha)[:12]}")
PY
}

leg_manifest_by_job() {  # <arm> <jobid> -> the manifest THAT job published
  local arm="$1" jid="$2" f
  local -a hits=()
  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_*_jid"${jid}"_manifest.txt; do
    [ -f "$f" ] && hits+=("$f")
  done
  [ "${#hits[@]}" -eq 1 ] || return 1
  printf '%s\n' "${hits[0]}"
}

leg_manifest_for() {   # <arm> <chunk_end> -> the ONE launcher manifest for that chunk
  local arm="$1" want="$2" f ce
  local -a hits=()
  for f in "$EXPDIR"/fa_orbit_*_"${arm}"_8x8_jid*_manifest.txt; do
    [ -f "$f" ] || continue
    ce="$(awk '/^chunk_end /{print $2; exit}' "$f" 2>/dev/null)"
    [ "$ce" = "$want" ] && hits+=("$f")
  done
  [ "${#hits[@]}" -eq 1 ] || return 1
  printf '%s\n' "${hits[0]}"
}

manifest_chunk_end() { # <manifest> -> its chunk_end line
  awk '/^chunk_end /{print $2; exit}' "$1" 2>/dev/null
}

process_arm() {
  local arm="$1"
  if [ -n "${ST[frozen_$arm]:-}" ]; then
    log "${arm}: FROZEN (${ST[frozen_$arm]}) — skipping; clear it in ${STATE} to resume"
    return 0
  fi

  # B2(b): a FAILING squeue is UNKNOWN, never "the queue is empty". Reading a
  # scheduler hiccup as "no live job" is how a running leg gets a twin.
  #
  # Round-5 r2 (blocking 1): this query is a FAST-PATH SKIP and a settlement
  # trigger, not the anti-duplicate guard. The guard is inside fa_orbit_submit.sh,
  # which holds a per-arm flock and repeats this query INSIDE it before sbatch —
  # the only place where "no live job" and "submit" are atomic with respect to
  # every other submitter, including a human at a shell.
  local queued qrc
  queued="$(squeue -h -u "$USER" -n "exp11-${arm}-train" -o '%i %T' 2>/dev/null)"; qrc=$?
  if [ "$qrc" -ne 0 ]; then
    log "${arm}: squeue failed (rc=${qrc}) — the queue state is UNKNOWN, skipping this poll (an unknown queue is never read as an empty one)"
    return 0
  fi

  # B3: settle the leg we last submitted — but only once the SCHEDULER says it
  # is really over. An empty or lagging sacct is not a verdict: the arm is left
  # alone (lastjob_<ARM> retained) so the same boundary cannot be resubmitted
  # while the previous attempt is still settling.
  local last="${ST[lastjob_$arm]:-}" st settled_job="" settled_fail=""
  if [ -n "$last" ] && [ -z "$queued" ]; then
    st="$(sacct -X -n -P -j "$last" -o State 2>/dev/null | head -1)"
    st="${st%%$'\n'*}"
    case "$st" in
      COMPLETED)
        clear_fail "$arm" "job ${last} COMPLETED"
        settled_job="$last"; unset "ST[lastjob_$arm]" ;;
      CANCELLED*)
        # A cancellation is a HUMAN acting on this run. Replacing the job the
        # operator just killed is the one thing the watchdog must never do.
        settled_job="$last"; unset "ST[lastjob_$arm]"
        freeze_now "$arm" "job ${last} was cancelled by operator — human intervention signal, not a retryable failure"
        return 0 ;;
      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED)
        settled_job="$last"; settled_fail=1; unset "ST[lastjob_$arm]"
        bump_fail "$arm" "job ${last} ended ${st}" ;;
      "")
        log "${arm}: sacct reports nothing yet for job ${last} — NOT settled, skipping this poll (lastjob_${arm} retained)"
        return 0 ;;
      *)
        log "${arm}: job ${last} is ${st}, which is not a terminal state — NOT settled, skipping this poll (lastjob_${arm} retained)"
        return 0 ;;
    esac
    [ -n "${ST[frozen_$arm]:-}" ] && return 0
  fi

  if [ -n "$queued" ]; then
    log "${arm}: live job (${queued//$'\n'/; }) — nothing to do"
    return 0
  fi

  local ckdir="${OUTPUT_ROOT}/exp11_${arm}/FLAC_exp11_${arm}/exp11_${arm}/checkpoints"
  local S; S="$(newest_ckpt_step "$ckdir")"
  if [ -z "$S" ]; then
    log "${arm}: no epoch=*-step=*.ckpt under ${ckdir} — nothing to resume, skipping"
    return 0
  fi
  # B4: a checkpoint PAST the target is an impossible state for a chunk chain
  # (every leg stops ON a boundary <= TARGET). It is not success — freeze.
  if [ "$S" -gt "$TARGET" ]; then
    freeze_now "$arm" "newest checkpoint is step ${S}, PAST the target ${TARGET} — impossible for a chunk chain; a human must explain this checkpoint before the chain continues"
    return 0
  fi

  # B4: a chunk that finished but was never recorded must be recorded FIRST —
  # BEFORE the DONE test, or the terminal chunk (e.g. 97500 -> 100000) would be
  # declared done and never recorded. The next chunk's preflight binds to the
  # recorded chain, not to whatever is on disk.
  local tip
  if [ "$S" -gt "$ANCHOR_STEP" ]; then
    tip="$(chain_tip_step "$arm")"
    is_num "$tip" || { log "${arm}: could not read the chain tip from ${REGISTRY}"; bump_fail "$arm" "registry unreadable"; return 0; }
    if [ "$tip" -lt "$S" ]; then
      local man ce
      # Round-5 r3 blocking 2: a FAILED leg may have SAVED its boundary
      # checkpoint before dying (wall-kill or class-7 after the save). That
      # file is STALE LINEAGE — its job was not COMPLETED, so it is neither
      # recorded nor resumed. The retry resumes the RECORDED tip at the SAME
      # boundary; Lightning versions the retry's save (-v1), and settlement by
      # job id later records the retry's own attested file. Exactly ONE
      # failure is counted for the failed job (the settle bump above).
      if [ -n "$settled_job" ] && [ "$settled_fail" = "1" ]; then
        log "${arm}: newest checkpoint (step ${S}) was left by FAILED job ${settled_job} — stale lineage, not recording and not resuming it; retrying the boundary from the recorded tip (${tip})"
        S="$tip"
      else
        if [ -n "$settled_job" ]; then
          if ! man="$(leg_manifest_by_job "$arm" "$settled_job")"; then
            log "${arm}: job ${settled_job} published no single manifest in ${EXPDIR} (zero or ambiguous) — cannot record the finished chunk"
            bump_fail "$arm" "no unique manifest for settled job ${settled_job}"
            return 0
          fi
          ce="$(manifest_chunk_end "$man")"
          if [ "$ce" != "$S" ]; then
            log "${arm}: job ${settled_job}'s manifest declares chunk_end ${ce:-<none>}, but the newest checkpoint on disk is step ${S} — refusing to attribute that checkpoint to this job"
            bump_fail "$arm" "manifest chunk_end ${ce:-<none>} != on-disk step ${S}"
            return 0
          fi
        elif ! man="$(leg_manifest_for "$arm" "$S")"; then
          # no settled job id in state (e.g. the watchdog was restarted with a
          # clean state file): fall back to UNIQUE chunk_end matching, and refuse
          # the moment it is ambiguous. If the unknown producer job had FAILED,
          # the recorder's sacct gate refuses and the arm freezes for a human —
          # the operator-restarted-into-a-mess case is deliberately manual.
          log "${arm}: no single launcher manifest with chunk_end ${S} in ${EXPDIR} (zero or ambiguous) and no settled job id in the state — cannot record the finished chunk"
          bump_fail "$arm" "manifest for chunk_end ${S} not uniquely identifiable"
          return 0
        fi
        local -a rec=("$RECORDER" "$arm" "$man" --registry "$REGISTRY")
        [ "$DRYRUN" = "1" ] && rec+=(--dry-run)
        log "${arm}: recording finished chunk ${tip} -> ${S} from $(basename "$man")"
        local out rc
        out="$("$PY" "${rec[@]}" 2>&1)"; rc=$?
        if [ "$rc" -ne 0 ]; then
          log "${arm}: RECORDER REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
          bump_fail "$arm" "recorder rc=${rc}"
          return 0
        fi
        clear_fail "$arm" "chunk ${tip} -> ${S} recorded"
        if [ "$DRYRUN" = "1" ]; then
          log "${arm}: DRYRUN — the chunk link was validated but not written, so the next chunk is not yet admissible"
        else
          tip="$(chain_tip_step "$arm")"
        fi
      fi
    fi
  fi

  # B4: DONE means BOTH the disk and the RECORD reached the target. A target
  # reached on disk but missing from the chain is an unfinished lineage, not a
  # finished campaign.
  # NON-BLOCKING adoption (round-5 r2): done_<ARM> is qualified BY TARGET. A
  # state file reused with a larger TARGET used to keep its old `done` entry, so
  # the arm was counted as finished and the watchdog exited one leg later.
  if [ "$S" -eq "$TARGET" ]; then
    tip="$(chain_tip_step "$arm")"
    if [ "$tip" = "$TARGET" ]; then
      if [ "${ST[done_$arm]:-}" != "${S}@${TARGET}" ]; then
        ST["done_$arm"]="${S}@${TARGET}"
        log "${arm}: DONE — newest checkpoint is step ${S} == target ${TARGET}, and the recorded chain tip is ${tip}"
      fi
      return 0
    fi
    log "${arm}: newest checkpoint is the target ${TARGET} but the recorded chain tip is ${tip} — NOT declaring DONE until the terminal chunk is recorded"
    [ "$DRYRUN" = "1" ] || bump_fail "$arm" "terminal chunk at ${TARGET} is not recorded"
    return 0
  fi

  # --- which file the next chunk resumes ------------------------------------
  # B3 (round-5 r2): above the anchor the resume file is the RECORDED tip's
  # final_ckpt_path, never a glob. Lightning's version counter means a boundary
  # can hold both `...-step=N.ckpt` (a failed attempt that saved) and
  # `...-step=N-v1.ckpt` (the successful retry); only the record knows which one
  # the recorded chunk actually produced. The unique-file fallback survives for
  # exactly one case — resuming the audited 40k anchor, which has no chain link.
  local step; step="$(chunk_for "$arm")"
  local next=$(( S + step ))
  [ "$next" -gt "$TARGET" ] && next="$TARGET"
  local resume=""
  if [ "$S" -gt "$ANCHOR_STEP" ]; then
    tip="$(chain_tip_step "$arm")"
    if [ "$tip" != "$S" ]; then
      log "${arm}: the newest checkpoint is step ${S} but the recorded chain tip is ${tip} — refusing to submit from an unrecorded checkpoint"
      bump_fail "$arm" "chain tip ${tip} != on-disk step ${S}"
      return 0
    fi
    resume="$(chain_tip_path "$arm")"
    if [ -z "$resume" ] || [ ! -f "$resume" ]; then
      log "${arm}: the recorded chain tip carries no usable final_ckpt_path (got '${resume:-<none>}') — the record, not the directory listing, names the file a chunk resumes"
      bump_fail "$arm" "recorded tip has no resumable final_ckpt_path"
      return 0
    fi
  else
    local -a ck=()
    mapfile -t ck < <(ckpt_files_at "$ckdir" "$S")
    if [ "${#ck[@]}" -ne 1 ]; then
      log "${arm}: expected exactly one checkpoint file at the anchor step ${S} in ${ckdir}, found ${#ck[@]}"
      bump_fail "$arm" "ambiguous checkpoint at step ${S}"
      return 0
    fi
    resume="${ck[0]}"
  fi

  # submit the next chunk through the ONLY sanctioned submitter, which owns the
  # anti-duplicate reservation (per-arm flock + in-lock queue re-check).
  log "${arm}: submitting chunk ${S} -> ${next} (chunk size ${step}): bash ${SUBMITTER} ${arm} --resume ${resume} --expected-step ${S} --chunk-end ${next} (DRYRUN=${DRYRUN})"
  local out rc jid
  out="$(DRYRUN="$DRYRUN" bash "$SUBMITTER" "$arm" --resume "$resume" \
          --expected-step "$S" --chunk-end "$next" 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ]; then
    log "${arm}: SUBMIT REFUSED (rc=${rc}): $(printf '%s' "$out" | head -4 | tr '\n' ' ')"
    bump_fail "$arm" "submit rc=${rc}"
    return 0
  fi
  jid="$(printf '%s' "$out" | awk '/^submitted /{print $NF}')"
  if [ -n "$jid" ]; then
    ST["lastjob_$arm"]="$jid"
    # Round-5 r2 (blocking 1(b)): persist NOW, not at end-of-pass. Between a
    # successful sbatch and the end of a poll the watchdog could die (or another
    # arm could abort the pass), and a lost lastjob_<ARM> means the next poll
    # settles nothing and re-submits this boundary.
    save_state || log "${arm}: WARNING — job ${jid} was submitted but the state file could not be written; record it by hand in ${STATE} as lastjob_${arm} ${jid}"
    log "${arm}: submitted chunk ${S} -> ${next} as job ${jid} (lastjob_${arm} persisted)"
  else
    log "${arm}: submitter returned 0 without a job id (DRYRUN=${DRYRUN}) — nothing queued"
  fi
  return 0
}

log "=== chunk watchdog start: arms ${ARM_LIST[*]} chunk ${CHUNK} (VANL ${CHUNK_VANL}) target ${TARGET} poll ${POLL}s max_consec_fail ${MAX_CONSEC_FAIL} oneshot ${ONESHOT} dryrun ${DRYRUN} pid $$ ==="

# --- B7: every selected arm must already carry an AUDITED anchor -------------
# VANL's registry row has neither final_ckpt_sha256 nor final_step, so its first
# chunk would be submitted, burn a queue slot, and be refused by the preflight.
# Anchoring is an OPERATOR action (fa_orbit_add_anchor.py audits the 40k
# checkpoint into the lineage) — the watchdog must never do it, so it freezes
# the arm here, before anything is submitted. Round-5 r2: the check now also
# proves step==40000, a well-formed digest, a unique anchor file in the
# registry's canonical directory, and that the file's bytes still hash to the
# audited value (one hash per arm, at startup only).
load_state
for ARM in "${ARM_LIST[@]}"; do
  [ -n "${ST[frozen_$ARM]:-}" ] && continue
  ANCHOR="$(anchor_state "$ARM")"
  case "$ANCHOR" in
    OK)         log "${ARM}: audited anchor verified at step ${ANCHOR_STEP} (registry sha matches the file on disk)" ;;
    MISSING)    freeze_now "$ARM" "no audited anchor — run fa_orbit_add_anchor.py ${ARM} first" ;;
    UNREADABLE) freeze_now "$ARM" "the registry row for ${ARM} could not be read from ${REGISTRY} — no audited anchor could be confirmed" ;;
    *)          freeze_now "$ARM" "the audited anchor did not verify: ${ANCHOR}" ;;
  esac
done
save_state || { echo "the state file ${STATE} could not be written — exiting rather than run with failure counters that do not persist" >&2; exit 3; }

while :; do
  load_state
  for ARM in "${ARM_LIST[@]}"; do
    process_arm "$ARM"
  done
  # NON-BLOCKING adoption: an unwritable state file means the next poll would
  # re-decide from stale state (re-submitting a boundary, losing a freeze). Stop.
  save_state || { log "FATAL: the state file ${STATE} could not be written — exiting rather than continue with unpersisted state"; exit 3; }

  REMAINING=0
  for ARM in "${ARM_LIST[@]}"; do
    # A done_<ARM> entry counts only for the TARGET it was recorded against;
    # `<step>@<target>` from an earlier, smaller campaign is ignored (and will be
    # recomputed by process_arm), so raising TARGET restarts the arm properly.
    DONE_AT="${ST[done_$ARM]:-}"
    [ -n "$DONE_AT" ] && [ "${DONE_AT#*@}" = "$TARGET" ] && continue
    [ -n "${ST[frozen_$ARM]:-}" ] && continue
    REMAINING=$((REMAINING + 1))
  done
  if [ "$REMAINING" -eq 0 ]; then
    log "=== every arm is DONE or FROZEN — watchdog exiting ==="
    break
  fi
  [ "$ONESHOT" = "1" ] && { log "=== ONESHOT: one pass complete, exiting ==="; break; }
  sleep "$POLL"
done
exit 0
