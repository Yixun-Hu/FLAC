#!/bin/bash
# ============================================================================
# exp_11 Q5 — batched-orbit qualification probe (1x L40, ~10 min).
#
# Runs fa_orbit_equiv_probe.py on the real DINOv3 stack:
#   * EVAL-mode equivalence (GATED): batched vs the legacy per-angle loop, fp32,
#     at B=8 for C in {4,8,16,32} plus the evaluation schedules B=64 and B=1.
#     Both a normwise-relative and a max-absolute bound must hold.
#   * TRAIN-mode qualification (RECORDED): grad-enabled, matmul precision
#     'medium', bf16-mixed — finite outputs and finite gradients are required and
#     the batched-vs-loop divergence is recorded for the disclosure record. That
#     divergence is the approved chunk-shared-RoPE recipe change, not an error.
#
# VRAM requalification is NOT done here: the batched peak comes from the 8x8 P0
# spot cells (C4L/C8/C16/C32), which exercise the real train path including
# backward and gradient-checkpoint recomputation through p0_runner.py.
#
# Fail-closed parity with the arm launcher: every provenance command's status is
# checked, both pipeline stages are captured, the pinned DINO snapshot/weight
# hash is gated inside the probe, and the result must be EXACTLY ONE strictly
# parsed EQUIVPROBE line carrying the expected config hash, the sample ids, the
# full cell set and verdict=PASS.
#
# Submit:
#   sbatch --job-name=exp11-equivprobe --gres=gpu:l40:1 --cpus-per-task=15 \
#          --mem=24G --time=00:10:00 \
#          --export=ALL,EXPECT_SHA=$(git rev-parse HEAD) \
#          worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_equiv_probe.sbatch
# ============================================================================
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/slurm_equivprobe_%x_%j.out

set -uo pipefail
REPO=/n/fs/gatrdp/codespace/FLAC
EXPDIR="$REPO/worklog/worklog_yixun/exp_11_fa_orbit_claude"
cd "$REPO" || exit 3
unset PYTHONPATH PYTHONOPTIMIZE
export PATH=/n/fs/gatrdp/envs/flac/bin:$PATH
export PYTHONNOUSERSITE=1
export HF_HOME=/n/fs/gatrdp/hf_cache
export HF_HUB_OFFLINE=1

die() { echo "$1"; exit "${2:-2}"; }

JID="${SLURM_JOB_ID:?must run under sbatch}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
: "${EXPECT_SHA:?EXPECT_SHA (full reviewed commit sha) must be exported at submission - abort}"
EXPECT_CELLS="${EXPECT_CELLS:-14}"      # len(expected_cells()): 4 + 8 + 2 (see the probe)
EXPECT_SAMPLES="${EXPECT_SAMPLES:-8}"
# Optional pin of the exact idx:relpath identities; when unset the wrapper still
# requires EXPECT_SAMPLES distinct idx:relpath tokens (NEW-3).
EXPECT_IDS="${EXPECT_IDS:-}"

echo "=== exp_11 batched-orbit qualification probe - ${TS} - job ${JID} - host $(hostname) ==="

# --- commit binding + drift gate; every provenance command's status is checked -
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)" || die "git rev-parse FAILED - abort"
[ -n "$HEAD_SHA" ] || die "git rev-parse returned nothing - abort"
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src eval_FLAC.py "$EXPDIR")" \
  || die "git status FAILED - refusing to treat an error as a clean tree - abort"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
echo "commit binding OK: ${HEAD_SHA}"

# --- shape + GPU asserts -----------------------------------------------------
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || die "expected 1 node, got ${SLURM_JOB_NUM_NODES} - abort"
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits) \
  || die "nvidia-smi query FAILED - abort"
[ "${#GPU_ROWS[@]}" -eq 1 ] || die "expected exactly 1 visible GPU, got ${#GPU_ROWS[@]} - abort"
GPU_NAME="$(echo "${GPU_ROWS[0]}" | cut -d, -f2 | sed 's/^ *//')"
case "$GPU_NAME" in *L40*) ;; *) die "allocated GPU '${GPU_NAME}' is not an L40 - abort";; esac
echo "GPU: ${GPU_NAME} $(echo "${GPU_ROWS[0]}" | cut -d, -f1)"

# --- runtime identity (checked, not informational) ---------------------------
python3 - <<'PY' || die "runtime version gate FAILED - abort"
import sys
import pytorch_lightning as pl, torch
bad = []
if pl.__version__ != "2.1.0":
    bad.append(f"pytorch_lightning {pl.__version__} != 2.1.0")
if torch.__version__ != "2.7.0+cu126":
    bad.append(f"torch {torch.__version__} != 2.7.0+cu126")
if bad:
    sys.exit("; ".join(bad))
print(f"runtime gate OK: pl {pl.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}")
PY

# --- the probe config must be the committed C32 arm manifest ------------------
PROBE_CFG="$EXPDIR/FLAC_AR_BF_C32.json"
[ -f "$PROBE_CFG" ] || die "missing ${PROBE_CFG} - abort"
CFG_SHA="$(sha256sum "$PROBE_CFG" | awk '{print $1}')" || die "sha256sum FAILED - abort"
[ -n "$CFG_SHA" ] || die "empty config sha256 - abort"
CFG_SHA12="${CFG_SHA:0:12}"
echo "config sha256: ${CFG_SHA} (expect cfg=${CFG_SHA12} in the result line)"

# --- BEGIN probe-run-and-verify (guard-tested by fa_orbit_equivprobe_wrapper_test.sh) --
# NEW-1: capture PIPESTATUS ATOMICALLY. `A="${PIPESTATUS[0]}"; B="${PIPESTATUS[1]}"`
# is two commands, and the first one replaces PIPESTATUS with its own one-element
# array -- under `set -u` the second expansion died with "PIPESTATUS[1]: unbound
# variable" (exit 127), so every strict check below used to be unreachable.
LOG="${LOG:-${EXPDIR}/fa_orbit_${TS}_jid${SLURM_JOB_ID}_equivprobe.log}"
python3 "$EXPDIR/fa_orbit_equiv_probe.py" --config "$PROBE_CFG" 2>&1 | tee "$LOG"
PIPE_RC=("${PIPESTATUS[@]}")
PROBE_RC="${PIPE_RC[0]}"; TEE_RC="${PIPE_RC[1]}"
[ "$TEE_RC" -eq 0 ] || die "tee exited ${TEE_RC} - the durable log is not trustworthy - abort" 7

# --- strict result parsing: exactly one line, and every field as expected -----
NLINES="$(grep -c '^EQUIVPROBE ' "$LOG")"
[ "$NLINES" -eq 1 ] || die "expected exactly 1 EQUIVPROBE line, found ${NLINES} - abort" 5
RESULT="$(grep '^EQUIVPROBE ' "$LOG")"
field() { echo "$RESULT" | tr ' ' '\n' | awk -F= -v k="$1" '$1==k{print $2; exit}'; }
GOT_CFG="$(field cfg)"; GOT_N="$(field nsamples)"; GOT_CELLS="$(field cells)"
GOT_IDS="$(field sample_ids)"; GOT_VERDICT="$(field verdict)"; GOT_PIN="$(field vit_pin)"
GOT_DEVICE="$(field device)"
[ "$GOT_CFG" = "$CFG_SHA12" ] || die "result cfg=${GOT_CFG} != ${CFG_SHA12} - abort" 5
[ "$GOT_N" = "$EXPECT_SAMPLES" ] || die "result nsamples=${GOT_N} != ${EXPECT_SAMPLES} - abort" 5
[ "$GOT_CELLS" = "${EXPECT_CELLS}/${EXPECT_CELLS}" ] || die "result cells=${GOT_CELLS} != ${EXPECT_CELLS}/${EXPECT_CELLS} - abort" 5
# NEW-2: a CPU/fp32 run must never be accepted as the bf16 qualification
[ "$GOT_DEVICE" = "cuda" ] || die "result device=${GOT_DEVICE} != cuda - the bf16 qualification requires a GPU - abort" 5
# NEW-3: the exact per-record identities, not merely a non-empty token
if [ -n "${EXPECT_IDS:-}" ]; then
  [ "$GOT_IDS" = "$EXPECT_IDS" ] || die "result sample_ids=${GOT_IDS} != the registered ${EXPECT_IDS} - abort" 5
else
  N_IDS="$(echo "$GOT_IDS" | tr ',' '\n' | grep -c ':')"
  [ "$N_IDS" -eq "$EXPECT_SAMPLES" ] || die "result sample_ids carries ${N_IDS} idx:relpath identities, expected ${EXPECT_SAMPLES} - abort" 5
  [ "$(echo "$GOT_IDS" | tr ',' '\n' | sort -u | wc -l)" -eq "$EXPECT_SAMPLES" ] || die "result sample_ids are not distinct: ${GOT_IDS} - abort" 5
fi
[ -n "$GOT_PIN" ] || die "result carries no DINO pin - abort" 5
[ "$GOT_VERDICT" = "PASS" ] || die "result verdict=${GOT_VERDICT} - abort" 4
[ "$PROBE_RC" -eq 0 ] || die "probe exited ${PROBE_RC} despite verdict=PASS - abort" "$PROBE_RC"

# the final classification is part of the tested chain: if it is not reached, the
# wrapper did not actually validate anything.
echo "=== equivalence probe PASSED (cfg ${GOT_CFG}, pin ${GOT_PIN}, device ${GOT_DEVICE}, ${GOT_CELLS} cells, ids ${GOT_IDS}) ==="
echo "log: ${LOG}"
# --- END probe-run-and-verify -------------------------------------------------

exit 0
