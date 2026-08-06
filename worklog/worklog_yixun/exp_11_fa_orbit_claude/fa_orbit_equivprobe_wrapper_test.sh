#!/usr/bin/env bash
# ============================================================================
# fa_orbit_equivprobe_wrapper_test.sh — guard test for the equivalence-probe
# wrapper's run-and-verify block.
#
# Re-review NEW-1: the wrapper could never succeed. It captured
#     PROBE_RC="${PIPESTATUS[0]}"; TEE_RC="${PIPESTATUS[1]}"
# but the FIRST assignment is itself a command, so it replaces PIPESTATUS with a
# one-element array; under `set -u` the second expansion dies with
# "PIPESTATUS[1]: unbound variable" (reproduced: exit 127). Every strict check
# after it was therefore unreachable — which is also why prior finding 5 could
# not be closed by inspection alone.
#
# This test EXTRACTS the real block from fa_orbit_equiv_probe.sbatch (between the
# BEGIN/END markers) and runs it against a stub `python3` that emits a canned
# probe log, so the wrapper's own text is exercised — not a copy of it. It
# asserts the happy path reaches the final classification with rc 0, and that
# every strict check actually fires on its own failure mode.
#
# Usage:  bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equivprobe_wrapper_test.sh
# Exit 0 = every case behaved as specified. No GPU, no Slurm, no probe run.
# ============================================================================
set -uo pipefail
cd "$(git -C "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)" || exit 3

EXPDIR="worklog/worklog_yixun/exp_11_fa_orbit_claude"
SBATCH="${EXPDIR}/fa_orbit_equiv_probe.sbatch"
[ -f "$SBATCH" ] || { echo "missing ${SBATCH} - abort"; exit 3; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

BLOCK="${TMP}/block.sh"
sed -n '/# --- BEGIN probe-run-and-verify/,/# --- END probe-run-and-verify/p' "$SBATCH" > "$BLOCK"
if ! [ -s "$BLOCK" ]; then
  echo "FAIL  could not extract the run-and-verify block (markers missing)"; exit 1
fi
echo "extracted $(wc -l < "$BLOCK") lines of the real wrapper block"

CFG_SHA12="abc123def456"
IDS="0:a.wav,1:b.wav,2:c.wav,3:d.wav,4:e.wav,5:f.wav,6:g.wav,7:h.wav"

# a canned probe stdout; $1 overrides fields
canned() {
  local verdict="${1:-PASS}" cells="${2:-14/14}" cfg="${3:-$CFG_SHA12}" \
        ids="${4:-$IDS}" nsamples="${5:-8}" device="${6:-cuda}"
  printf 'probe: warming up\n'
  printf 'EQUIVPROBE cfg=%s vit_pin=114c13799502/4610ad75edef device=%s nsamples=%s ' \
         "$cfg" "$device" "$nsamples"
  printf 'sample_ids=%s cells=%s gate_rel_norm=1.0e-08 gate_max_abs=2.0e-07 ' "$ids" "$cells"
  printf 'rec_rel_norm=3.0e-02 rec_max_abs=1.0e-01 bf16_rel_norm=4.0e-03 bf16_max_abs=2.0e-02 '
  printf 'tol_rel=1e-06 tol_abs=1e-05 verdict=%s\n' "$verdict"
}

# Run the extracted block with a stub python3 that prints $STUB_OUT and returns $STUB_RC.
run_block() {
  local stub_out="$1" stub_rc="${2:-0}"
  cat > "${TMP}/harness.sh" <<HARNESS
set -uo pipefail
EXPDIR="$(pwd)/${EXPDIR}"
PROBE_CFG="\${EXPDIR}/FLAC_AR_BF_C32.json"
LOG="${TMP}/probe.log"
CFG_SHA12="${CFG_SHA12}"
EXPECT_SAMPLES=8
EXPECT_CELLS=14
EXPECT_IDS="${IDS}"
TS=testts
SLURM_JOB_ID=999
die() { echo "\$1"; exit "\${2:-2}"; }
python3() { cat "${stub_out}"; return ${stub_rc}; }
HARNESS
  cat "$BLOCK" >> "${TMP}/harness.sh"
  bash "${TMP}/harness.sh" 2>&1
}

expect() {  # <name> <want-rc> <want-substring> <stub-out-file> [stub-rc]
  local name="$1" want_rc="$2" want_txt="$3" stub="$4" stub_rc="${5:-0}"
  local out rc
  out="$(run_block "$stub" "$stub_rc")"; rc=$?
  if [ "$rc" -eq "$want_rc" ] && echo "$out" | grep -qF -- "$want_txt"; then
    echo "PASS  ${name}  (rc=${rc})"; PASS=$((PASS + 1))
  else
    echo "FAIL  ${name}: want rc=${want_rc} + '${want_txt}', got rc=${rc}"
    echo "$out" | tail -4 | sed 's/^/        | /'; FAIL=$((FAIL + 1))
  fi
}

canned                                   > "${TMP}/ok.out"
canned FAIL                              > "${TMP}/verdict_fail.out"
canned PASS 13/14                        > "${TMP}/short_cells.out"
canned PASS 14/14 deadbeef0000           > "${TMP}/wrong_cfg.out"
canned PASS 14/14 "$CFG_SHA12" "0:x.wav" > "${TMP}/wrong_ids.out"
canned PASS 14/14 "$CFG_SHA12" "$IDS" 4  > "${TMP}/wrong_nsamples.out"
canned PASS 14/14 "$CFG_SHA12" "$IDS" 8 cpu > "${TMP}/cpu_device.out"
{ canned; canned; }                      > "${TMP}/two_lines.out"
printf 'probe crashed before the result\n' > "${TMP}/no_line.out"

echo "--- the happy path must REACH the final classification (NEW-1) ---"
expect "green probe passes end to end" 0 "equivalence probe PASSED" "${TMP}/ok.out"

echo "--- every strict check must actually fire (prior finding 5) ---"
expect "verdict=FAIL rejected"        4 "verdict=FAIL"        "${TMP}/verdict_fail.out"
expect "short cell set rejected"      5 "cells="              "${TMP}/short_cells.out"
expect "wrong config hash rejected"   5 "cfg="                "${TMP}/wrong_cfg.out"
expect "wrong sample ids rejected"    5 "sample_ids"          "${TMP}/wrong_ids.out"
expect "wrong sample count rejected"  5 "nsamples="           "${TMP}/wrong_nsamples.out"
expect "cpu device rejected"          5 "device="             "${TMP}/cpu_device.out"
expect "two result lines rejected"    5 "exactly 1"           "${TMP}/two_lines.out"
expect "missing result line rejected" 5 "exactly 1"           "${TMP}/no_line.out"
expect "probe nonzero rc rejected"    4 "verdict=FAIL"        "${TMP}/verdict_fail.out" 4

echo
echo "=== wrapper guard tests: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
