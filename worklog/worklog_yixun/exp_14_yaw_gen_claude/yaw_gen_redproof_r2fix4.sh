#!/usr/bin/env bash
# ============================================================================
# yaw_gen_redproof_r2fix4.sh — the RED evidence for r2-fix4 (review Y1/Z1), as a
# runnable artifact rather than a claim in a commit message.
#
# It checks the PRE-FIX blobs of the two submitters out of git into a temp dir and
# shows, on those copies, that an environment variable naming an executable was
# executed. The stand-ins are harmless recorders (/bin/echo-shaped scripts that
# only append a line), so the escape is demonstrated WITHOUT touching Slurm: what
# is proven is that the pre-fix code RAN what the environment named, which is the
# whole vulnerability — a stand-in that had been `sbatch` would have submitted.
#
# Then it runs the same probes against the CURRENT files and shows the refusals.
#
# Usage:  bash yaw_gen_redproof_r2fix4.sh [PRE_FIX_SHA]     (default: 2131cfb)
# Exit 0 = every pre-fix escape reproduced AND every post-fix refusal held.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
EXPDIR="worklog/worklog_yixun/exp_14_yaw_gen_claude"
PRE="${1:-2131cfb}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok()  { echo "PASS  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL  $1"; FAIL=$((FAIL + 1)); }

echo "=== r2-fix4 RED proof — pre-fix ${PRE} vs the working tree — $(date -Is) ==="
git show "${PRE}:${EXPDIR}/yaw_gen_submit_grid.sh"   > "${TMP}/pre_grid.sh"   || exit 3
git show "${PRE}:${EXPDIR}/yaw_gen_screen_submit.sh" > "${TMP}/pre_submit.sh" || exit 3
echo "pre-fix blobs: grid $(git rev-parse "${PRE}:${EXPDIR}/yaw_gen_submit_grid.sh" | cut -c1-12)" \
     "submit $(git rev-parse "${PRE}:${EXPDIR}/yaw_gen_screen_submit.sh" | cut -c1-12)"

# A harmless stand-in that records that it ran. Anything the pre-fix code will
# execute from an env var, it will execute — this one just leaves a trace.
STAND_IN="${TMP}/stand_in.sh"
printf '#!/usr/bin/env bash\nprintf "STAND-IN RAN AS %%s: %%s\\n" "$STAND_IN_ROLE" "$*" >> "%s"\nexit 0\n' \
  "${TMP}/ran.txt" > "$STAND_IN"
chmod +x "$STAND_IN"
printf '%s\n' "$(git rev-parse HEAD)" > "${TMP}/pin"
mkdir -p "${TMP}/out"

# --- (a) YAW_GEN_SQUEUE was EXEC'd in test mode ------------------------------
rm -f "${TMP}/ran.txt"
env STAND_IN_ROLE=squeue YAW_GEN_TEST_MODE=1 YAW_GEN_SQUEUE="$STAND_IN" \
    YAW_GEN_PIN_FILE="${TMP}/pin" YAW_GEN_COMMAND_LOG="${TMP}/cmd.md" \
    YAW_GEN_TEST_RECORD="${TMP}/rec.txt" OUTPUT_ROOT="${TMP}/out" \
    bash "${TMP}/pre_grid.sh" WAVE=vctl >/dev/null 2>&1
if grep -q "STAND-IN RAN AS squeue" "${TMP}/ran.txt" 2>/dev/null; then
  ok "PRE-FIX: YAW_GEN_SQUEUE was EXECUTED in test mode ($(grep -c . "${TMP}/ran.txt") call(s))"
  sed 's/^/        | /' "${TMP}/ran.txt" | head -2
else
  bad "PRE-FIX: could not reproduce the YAW_GEN_SQUEUE escape (fixture drift?)"
fi

# --- (b) YAW_GEN_SYNC was EXEC'd by the single-cell submitter ----------------
rm -f "${TMP}/ran.txt"
env STAND_IN_ROLE=sync DRYRUN=1 YAW_GEN_SYNC="$STAND_IN" YAW_GEN_PIN_FILE="${TMP}/pin" \
    bash "${TMP}/pre_submit.sh" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1
env STAND_IN_ROLE=sync YAW_GEN_TEST_MODE=1 YAW_GEN_SYNC="$STAND_IN" \
    YAW_GEN_TEST_RECORD="${TMP}/rec2.txt" YAW_GEN_PIN_FILE="${TMP}/pin" \
    YAW_GEN_INTENT_DIR="$TMP" bash "${TMP}/pre_submit.sh" ARM=C4L CELL=zref STEP=40000 \
    >/dev/null 2>&1
if grep -q "STAND-IN RAN AS sync" "${TMP}/ran.txt" 2>/dev/null; then
  ok "PRE-FIX: YAW_GEN_SYNC was EXECUTED by the single-cell submitter"
  sed 's/^/        | /' "${TMP}/ran.txt" | head -2
else
  bad "PRE-FIX: could not reproduce the YAW_GEN_SYNC escape (fixture drift?)"
fi

# --- (c) FA_ORBIT_SBATCH: any spelling but "sbatch" counted as a mock --------
# (closed one round earlier, in r2-fix3; kept here so the whole class is on file)
rm -f "${TMP}/ran.txt"
git show "2131cfb~1:${EXPDIR}/yaw_gen_screen_submit.sh" > "${TMP}/pre3_submit.sh" 2>/dev/null || true
if [ -s "${TMP}/pre3_submit.sh" ]; then
  env STAND_IN_ROLE=sbatch FA_ORBIT_SBATCH="$STAND_IN" FA_ORBIT_SCONTROL=/bin/true \
      FA_ORBIT_SCANCEL=/bin/true YAW_GEN_PIN_FILE="${TMP}/pin" YAW_GEN_INTENT_DIR="$TMP" \
      bash "${TMP}/pre3_submit.sh" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1
  if grep -q "STAND-IN RAN AS sbatch" "${TMP}/ran.txt" 2>/dev/null; then
    ok "PRE-FIX (r2-fix2): an arbitrary FA_ORBIT_SBATCH was EXECUTED as the submitter"
    sed 's/^/        | /' "${TMP}/ran.txt" | head -1
  else
    bad "PRE-FIX (r2-fix2): could not reproduce the FA_ORBIT_SBATCH escape"
  fi
fi

# --- POST-FIX: every one of those is refused ---------------------------------
echo "--- the same probes against the working tree ---"
for PROBE in "YAW_GEN_SQUEUE=${STAND_IN}" "YAW_GEN_SYNC=${STAND_IN}" "YAW_GEN_FOO=1"; do
  rm -f "${TMP}/ran.txt"
  out="$(env STAND_IN_ROLE=probe YAW_GEN_TEST_MODE=1 "$PROBE" \
         YAW_GEN_PIN_FILE="${TMP}/pin" OUTPUT_ROOT="${TMP}/out" \
         bash "${EXPDIR}/yaw_gen_submit_grid.sh" WAVE=vctl 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist" \
     && [ ! -f "${TMP}/ran.txt" ]; then
    ok "POST-FIX: ${PROBE%%=*} is refused by the entry allowlist and never executed"
  else
    bad "POST-FIX: ${PROBE%%=*} was tolerated (rc=${rc})"
  fi
done
for PROBE in "FA_ORBIT_SBATCH=${STAND_IN}" "YAW_GEN_SYNC=${STAND_IN}"; do
  rm -f "${TMP}/ran.txt"
  out="$(env STAND_IN_ROLE=probe "$PROBE" bash "${EXPDIR}/yaw_gen_screen_submit.sh" \
         ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist" \
     && [ ! -f "${TMP}/ran.txt" ]; then
    ok "POST-FIX: ${PROBE%%=*} is refused by the submitter and never executed"
  else
    bad "POST-FIX: ${PROBE%%=*} was tolerated (rc=${rc})"
  fi
done

echo "=== red proof: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
