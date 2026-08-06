#!/bin/bash
# ============================================================================
# exp_11 Q5 — batched-orbit equivalence probe (1x L40, ~10 min).
#
# Proves on the REAL DINOv3 conditioner stack and real AcousticRooms samples that
# the batched orbit execution reproduces the pre-batching loop, in fp32 and under
# bf16 autocast, for C4 and C32. Pre-registered tolerances live in the probe
# (fp32 rel <= 1e-6, bf16 rel <= 2e-3); it emits one EQUIVPROBE line and exits 4
# if either is exceeded.
#
# No training, no checkpoints, no writes outside this folder: it builds the model
# from FLAC_AR_BF_C32.json with the pinned pretrained ViT (HF_HUB_OFFLINE=1) and
# reads the first few items of the train split.
#
# Submit (gates mirror p0_profile.sbatch):
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

JID="${SLURM_JOB_ID:?must run under sbatch}"
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
: "${EXPECT_SHA:?EXPECT_SHA (full reviewed commit sha) must be exported at submission - abort}"

echo "=== exp_11 batched-orbit equivalence probe - ${TS} - job ${JID} - host $(hostname) ==="

# --- commit binding + drift gate (same surfaces as p0_profile.sbatch) --------
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || { echo "HEAD ${HEAD_SHA} != EXPECT_SHA ${EXPECT_SHA} - abort"; exit 2; }
DRIFT="$(git status --porcelain --untracked-files=no -- train.py defaults.ini src "$EXPDIR" 2>/dev/null)"
[ -z "$DRIFT" ] || { echo "tracked measurement surfaces modified since review - abort:"; echo "$DRIFT"; exit 2; }
echo "commit binding OK: ${HEAD_SHA}"

# --- shape + GPU asserts -----------------------------------------------------
[ "${SLURM_JOB_NUM_NODES:-1}" = "1" ] || { echo "expected 1 node - abort"; exit 2; }
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader,nounits)
[ "${#GPU_ROWS[@]}" -eq 1 ] || { echo "expected exactly 1 visible GPU, got ${#GPU_ROWS[@]} - abort"; exit 2; }
GPU_NAME="$(echo "${GPU_ROWS[0]}" | cut -d, -f2 | sed 's/^ *//')"
case "$GPU_NAME" in *L40*) ;; *) echo "allocated GPU '${GPU_NAME}' is not an L40 - abort"; exit 2;; esac
echo "GPU: ${GPU_NAME} $(echo "${GPU_ROWS[0]}" | cut -d, -f1)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- the probe config must be the committed C32 arm manifest ------------------
PROBE_CFG="$EXPDIR/FLAC_AR_BF_C32.json"
[ -f "$PROBE_CFG" ] || { echo "missing ${PROBE_CFG} - abort"; exit 2; }
echo "config sha256: $(sha256sum "$PROBE_CFG" | awk '{print $1}')"

LOG="${EXPDIR}/fa_orbit_${TS}_jid${JID}_equivprobe.log"
python3 "$EXPDIR/fa_orbit_equiv_probe.py" --config "$PROBE_CFG" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
grep -q "^EQUIVPROBE " "$LOG" || { echo "probe produced no EQUIVPROBE line - abort"; exit 5; }
echo "=== equivalence probe exited rc=${rc} (log ${LOG}) ==="
exit "$rc"
