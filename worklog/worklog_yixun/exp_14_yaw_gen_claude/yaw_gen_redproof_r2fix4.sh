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
# CONTAINMENT (review W4): the pre-fix copies are retargeted with sed so their
# MAIN_REPO is a temporary directory, and the shared store helper they call is a
# stub this script writes. An earlier version ran them against the real repo and
# left one stale lease in exp_11's campaign store; that cannot happen here — the
# probes have no path to /n/fs/gatrdp/codespace/FLAC/.measure_worktrees at all.
#
# Then it runs the same probes against the CURRENT files and shows the refusals.
#
# THREAT MODEL (Planner, 2026-08-11): accident prevention, not adversarial shell
# environments. Every probe here — pre-fix AND post-fix — runs against COPIES of
# the kit retargeted at a temporary MAIN_REPO, so no probe can touch the real
# repository or the shared measure-worktree store, and the proof verifies that at
# the end by comparing the store's whole lease listing before and after.
#
# Usage:  bash yaw_gen_redproof_r2fix4.sh [PRE_FIX_SHA]     (default: 2131cfb)
# Exit 0 = every pre-fix escape reproduced AND every post-fix refusal held.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3
EXPDIR="worklog/worklog_yixun/exp_14_yaw_gen_claude"
PRE="${1:-2131cfb}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0; START_EPOCH="$(date +%s)"
# The real store's complete lease listing, before anything runs. Comparing this
# (rather than looking for one job id) is what makes "we touched nothing" a
# statement about the whole store.
REAL_STORE=/n/fs/gatrdp/codespace/FLAC/.measure_worktrees
STORE_BEFORE="$(ls -1 "$REAL_STORE"/*/.leases/* 2>/dev/null | sort)"
ok()  { echo "PASS  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL  $1"; FAIL=$((FAIL + 1)); }

echo "=== r2-fix4 RED proof — pre-fix ${PRE} vs the working tree — $(date -Is) ==="
# --- containment: a temporary MAIN_REPO and a stub store helper -------------
FAKE="${TMP}/fake_repo"
FAKE_EXP="${FAKE}/worklog/worklog_yixun/exp_14_yaw_gen_claude"
FAKE_E11="${FAKE}/worklog/worklog_yixun/exp_11_fa_orbit_claude"
mkdir -p "$FAKE_EXP" "$FAKE_E11" "${FAKE}/.measure_worktrees" "${FAKE}/.leases"
# a REAL (empty) git repository, so the pre-fix code's own commit checks pass and
# the probes reach the call under test — without any path to the real repo
git -c init.defaultBranch=main init -q "$FAKE"
git -C "$FAKE" -c user.email=redproof@local -c user.name=redproof \
    commit -q --allow-empty -m "redproof containment root"
# the stub the pre-fix submitter will call instead of the real store helper: it
# hands back a temp directory and records leases there, so no real worktree, no
# real lease and no real lock are ever touched.
cat > "${FAKE_E11}/fa_orbit_measure_worktree.sh" <<STUB
#!/usr/bin/env bash
case "\${1:-}" in
  --with-lock) shift
               exec 8>"${FAKE}/.measure_worktrees/.store.lock"
               FA_ORBIT_STORE_LOCK_HELD=1 exec "\$@" ;;
  --pinned)    echo "<none>"; exit 1 ;;
  --frozen)    exit 0 ;;
  --lease)     mkdir -p "${FAKE}/.leases"; printf 'jobid %s\n' "\$2" > "${FAKE}/.leases/\$2"; exit 0 ;;
  --release)   rm -f "${FAKE}/.leases/\$2"; exit 0 ;;
  *)           echo "${FAKE}"; exit 0 ;;
esac
STUB
chmod +x "${FAKE_E11}/fa_orbit_measure_worktree.sh"
: > "${FAKE}/.measure_worktrees/.campaign_freeze"
retarget() {   # <sha> <basename> -> a pre-fix copy whose MAIN_REPO is the temp root
  git show "${1}:${EXPDIR}/${2}" \
    | sed "s|^MAIN_REPO=/n/fs/gatrdp/codespace/FLAC$|MAIN_REPO=${FAKE}|" > "${FAKE_EXP}/${2}"
  [ -s "${FAKE_EXP}/${2}" ] || return 1
  grep -q "^MAIN_REPO=${FAKE}$" "${FAKE_EXP}/${2}"
}
retarget "$PRE" yaw_gen_submit_grid.sh    || { echo "cannot retarget the pre-fix grid"; exit 3; }
retarget "$PRE" yaw_gen_screen_submit.sh  || { echo "cannot retarget the pre-fix submitter"; exit 3; }
cp "${EXPDIR}/exp14_validate_cell.py" "${EXPDIR}/exp14_ckpt_expect.json" "$FAKE_EXP/"
cp "${FAKE_EXP}/yaw_gen_submit_grid.sh"   "${TMP}/pre_grid.sh"
cp "${FAKE_EXP}/yaw_gen_screen_submit.sh" "${TMP}/pre_submit.sh"
echo "containment: MAIN_REPO=${FAKE} (stub store helper; the real store is unreachable)"
echo "pre-fix blobs: grid $(git rev-parse "${PRE}:${EXPDIR}/yaw_gen_submit_grid.sh" | cut -c1-12)" \
     "submit $(git rev-parse "${PRE}:${EXPDIR}/yaw_gen_screen_submit.sh" | cut -c1-12)"

# A harmless stand-in that records that it ran. Anything the pre-fix code will
# execute from an env var, it will execute — this one just leaves a trace.
STAND_IN="${TMP}/stand_in.sh"
printf '#!/usr/bin/env bash\nprintf "STAND-IN RAN AS %%s: %%s\\n" "$STAND_IN_ROLE" "$*" >> "%s"\nexit 0\n' \
  "${TMP}/ran.txt" > "$STAND_IN"
chmod +x "$STAND_IN"
printf '%s\n' "$(git -C "$FAKE" rev-parse HEAD)" > "${TMP}/pin"
mkdir -p "${TMP}/out"

# --- (a) YAW_GEN_SQUEUE was EXEC'd in test mode ------------------------------
rm -f "${TMP}/ran.txt"
timeout 60 env STAND_IN_ROLE=squeue YAW_GEN_TEST_MODE=1 YAW_GEN_SQUEUE="$STAND_IN" \
    YAW_GEN_PIN_FILE="${TMP}/pin" YAW_GEN_COMMAND_LOG="${TMP}/cmd.md" \
    YAW_GEN_TEST_RECORD="${TMP}/rec.txt" OUTPUT_ROOT="${TMP}/out" \
    bash "${FAKE_EXP}/yaw_gen_submit_grid.sh" WAVE=vctl >/dev/null 2>&1
if grep -q "STAND-IN RAN AS squeue" "${TMP}/ran.txt" 2>/dev/null; then
  ok "PRE-FIX: YAW_GEN_SQUEUE was EXECUTED in test mode ($(grep -c . "${TMP}/ran.txt") call(s))"
  sed 's/^/        | /' "${TMP}/ran.txt" | head -2
else
  bad "PRE-FIX: could not reproduce the YAW_GEN_SQUEUE escape (fixture drift?)"
fi

# --- (b) YAW_GEN_SYNC was EXEC'd by the single-cell submitter ----------------
rm -f "${TMP}/ran.txt"
timeout 60 env STAND_IN_ROLE=sync DRYRUN=1 YAW_GEN_SYNC="$STAND_IN" YAW_GEN_PIN_FILE="${TMP}/pin" \
    bash "${FAKE_EXP}/yaw_gen_screen_submit.sh" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1
timeout 60 env STAND_IN_ROLE=sync YAW_GEN_TEST_MODE=1 YAW_GEN_SYNC="$STAND_IN" \
    YAW_GEN_TEST_RECORD="${TMP}/rec2.txt" YAW_GEN_PIN_FILE="${TMP}/pin" \
    YAW_GEN_INTENT_DIR="$TMP" bash "${FAKE_EXP}/yaw_gen_screen_submit.sh" ARM=C4L CELL=zref STEP=40000 \
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
retarget "2131cfb~1" yaw_gen_screen_submit.sh 2>/dev/null \
  && cp "${FAKE_EXP}/yaw_gen_screen_submit.sh" "${TMP}/pre3_submit.sh" || true
if [ -s "${TMP}/pre3_submit.sh" ]; then
  env STAND_IN_ROLE=sbatch FA_ORBIT_SBATCH="$STAND_IN" FA_ORBIT_SCONTROL=/bin/true \
      FA_ORBIT_SCANCEL=/bin/true YAW_GEN_PIN_FILE="${TMP}/pin" YAW_GEN_INTENT_DIR="$TMP" \
      bash "${FAKE_EXP}/yaw_gen_screen_submit.sh" ARM=C4L CELL=zref STEP=40000 >/dev/null 2>&1
  if grep -q "STAND-IN RAN AS sbatch" "${TMP}/ran.txt" 2>/dev/null; then
    ok "PRE-FIX (r2-fix2): an arbitrary FA_ORBIT_SBATCH was EXECUTED as the submitter"
    sed 's/^/        | /' "${TMP}/ran.txt" | head -1
  else
    bad "PRE-FIX (r2-fix2): could not reproduce the FA_ORBIT_SBATCH escape"
  fi
fi

# --- POST-FIX: every one of those is refused ---------------------------------
# The CURRENT kit, retargeted the same way: a post-fix probe must not be the one
# thing in this proof that runs the real submitter (review V2). The refusals it
# asserts are produced by the entry allowlist, which is identical in the copy.
CUR_EXP="${FAKE}/current"; mkdir -p "$CUR_EXP"
for f in yaw_gen_submit_grid.sh yaw_gen_screen_submit.sh; do
  sed "s|^MAIN_REPO=/n/fs/gatrdp/codespace/FLAC$|MAIN_REPO=${FAKE}|" "${EXPDIR}/${f}" > "${CUR_EXP}/${f}"
done
cp "${EXPDIR}/exp14_validate_cell.py" "${EXPDIR}/exp14_ckpt_expect.json" "$CUR_EXP/"
echo "--- the same probes against the working tree (retargeted copies) ---"
for PROBE in "YAW_GEN_SQUEUE=${STAND_IN}" "YAW_GEN_SYNC=${STAND_IN}" "YAW_GEN_FOO=1"; do
  rm -f "${TMP}/ran.txt"
  out="$(timeout 60 env STAND_IN_ROLE=probe YAW_GEN_TEST_MODE=1 "$PROBE" \
         YAW_GEN_PIN_FILE="${TMP}/pin" OUTPUT_ROOT="${TMP}/out" \
         bash "${CUR_EXP}/yaw_gen_submit_grid.sh" WAVE=vctl 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist" \
     && [ ! -f "${TMP}/ran.txt" ]; then
    ok "POST-FIX: ${PROBE%%=*} is refused by the entry allowlist and never executed"
  else
    bad "POST-FIX: ${PROBE%%=*} was tolerated (rc=${rc})"
  fi
done
for PROBE in "FA_ORBIT_SBATCH=${STAND_IN}" "YAW_GEN_SYNC=${STAND_IN}"; do
  rm -f "${TMP}/ran.txt"
  out="$(timeout 60 env STAND_IN_ROLE=probe "$PROBE" bash "${CUR_EXP}/yaw_gen_screen_submit.sh" \
         ARM=C4L CELL=zref STEP=40000 2>&1)"; rc=$?
  if [ "$rc" -ne 0 ] && echo "$out" | grep -q "not on this mode's allowlist" \
     && [ ! -f "${TMP}/ran.txt" ]; then
    ok "POST-FIX: ${PROBE%%=*} is refused by the submitter and never executed"
  else
    bad "POST-FIX: ${PROBE%%=*} was tolerated (rc=${rc})"
  fi
done

# --- containment, verified ---------------------------------------------------
STORE_AFTER="$(ls -1 "$REAL_STORE"/*/.leases/* 2>/dev/null | sort)"
if [ "$STORE_BEFORE" = "$STORE_AFTER" ]; then
  ok "CONTAINMENT: the real store's ENTIRE lease listing is unchanged ($(printf '%s' "$STORE_AFTER" | grep -c . || true) lease(s))"
else
  bad "CONTAINMENT: the real store's lease listing changed"
  diff <(printf '%s\n' "$STORE_BEFORE") <(printf '%s\n' "$STORE_AFTER") | sed 's/^/        | /' | head -5
fi
if [ -n "$(ls -A "${FAKE}/.leases" 2>/dev/null)" ]; then
  ok "...the pre-fix probes' leases landed in the TEMP store instead ($(ls "${FAKE}/.leases" | tr '\n' ' '))"
else
  ok "...the pre-fix probes wrote no lease at all"
fi

echo "=== red proof: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
