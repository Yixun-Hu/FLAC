#!/usr/bin/env bash
# EXP03 DIFF 1/19: exp_03 (exp03n) MAX-POOL + MLP conditioning head. This file is exp09_launch.sh with 19 marked non-scientific diffs (absolute cluster roots + PYTHONPATH, log path, frozen-VRAM file, gate script, model config, run names, save-dir, REQUIRED pins, netrc identity gate, banners). EVERY scientific flag below is byte-identical to exp-09's reviewed launcher; the ONLY model delta is cond_pool=max_mlp in FLAC_AR_exp03n.json.
# ============================================================================
# exp09_launch.sh - exp-09 (FLAC no-SSL cylindrical DINOv3): C2 full training,
# 2-GPU DDP + SyncBN. B-F/P1 protocol EXACTLY (micro 32/GPU x2 x accum1 = eff 64,
# SyncBN, seed 42, AdamW 5e-5/wd 1e-3/InverseLR(1e6,0.5,0.99), EMA, bf16-mixed,
# 67,500 steps, ckpt every 2,500) - the ONLY differences from B-F are the ViT
# backbone (cylindrical_dinov3 + gauge=cylindrical_xyz, official weights, NO SSL)
# and cond_method fa_invariant with frame_avg_angles [0.0] (one base/frame pass,
# no extra frame-average passes), PLUS the exp_03 max-pool + MLP conditioning head. See FLAC_AR_exp03n.json and its per-variant config-contract test.   # EXP03 DIFF 14/19: arm description
#
# Cloned from exp_07's bf_scratch_launch.sh: same reviewed skeleton (fail-closed
# free-VRAM gate, wandb identity gate, pre-launch pin gate, `set -uo pipefail`).
#
# GPU-GATED (plan §6): B-F completion does NOT authorize this launch. C1/C2/D each
# need an explicit Yixun go AND a free-VRAM check. THIS launcher (C2/full) BINDS the
# FROZEN threshold MIN_FREE_MB (= measured exp03n L40 peak x 1.15) that exp03n_probe.sh   # EXP03 DIFF 15/19: threshold provenance
# derived, read from the records file exp03n_frozen_min_free.txt: absent file, non-numeric   # EXP03 DIFF 17/19: records file name
# value, or a MIN_FREE_MB override != the frozen value all make this script REFUSE
# (fail-closed). It can no longer launch against B-F's inherited number or an arbitrary
# override. Optional resume via CKPT_PATH (--ckpt-path passthrough; see below).
# ============================================================================
set -uo pipefail
cd /n/fs/gatrdp/codespace/exp03-maxpool-mlp-cond || exit 3   # EXP03 DIFF 2/19: absolute worktree root for THIS cluster (handoff §3.2 hardcoded-root trap; the launcher is invoked from a Slurm wrapper in the cylindrical repo)
export PYTHONPATH=/n/fs/gatrdp/codespace/cylindrical-dinov3/src   # EXP03 DIFF 3/19: process-local package src (reviewed convention; nothing installed into the conda env)

EXPDIR="worklog/worklog_yixun/exp_09_cyl_no_ssl"
LOGGER="${LOGGER:-wandb}"
MB="${MB:-32}"    # micro-batch per GPU (B-F/P1 BN-64 rung)
ACC="${ACC:-1}"   # accumulation      (B-F/P1 BN-64 rung)
CKPT_PATH="${CKPT_PATH:-}"   # optional resume checkpoint (train.py:230; documented below)
TS="$(date '+%Y-%m-%d_%H-%M-%S')"
# The train log is a C2 records artifact kept in EXPDIR by default; EXP09_LOG_DIR relocates
# it (e.g. for CPU tests) so a gate rehearsal never dirties the worktree.
LOG="${EXP09_LOG_DIR:-/n/fs/gatrdp/codespace/cylindrical-dinov3/worklog/worklog_yixun_neuronic/exp_03_maxpool_mlp_cond_claude}/maxpool_mlp_cond_${TS}_j${SLURM_JOB_ID:-nojob}_train.log"   # EXP03 DIFF 4/19: teed log into the exp_03 records folder, job-id stamped

# invariant (review findings): accumulation never feeds BN statistics, so the
# BN=64 mandate leaves exactly ONE legal rung - pin it literally (string equality;
# arithmetic like MB*2*ACC==64 is bypassable via bash int overflow):
[ "$MB" = "32" ] && [ "$ACC" = "1" ] || { echo "only the BN-compliant rung MB=32 ACC=1 is allowed (got MB='${MB}' ACC='${ACC}') - abort"; exit 2; }

# --- EXACT-MATCH frozen-threshold binding (integrative-review findings 3 & 5) --------------
# The launcher must NOT accept an arbitrary numeric MIN_FREE_MB (the old gate did, so a value
# BELOW the derived threshold could launch). Instead it BINDS to the exact value the C1 fit
# probe FROZE into a records file, and cross-checks any provided MIN_FREE_MB against it:
#   * frozen-records file absent            -> REFUSE (C1 has not frozen the threshold yet);
#   * frozen value non-numeric / empty      -> REFUSE;
#   * MIN_FREE_MB provided and != frozen    -> REFUSE (arbitrary override rejected);
#   * MIN_FREE_MB absent, or == frozen      -> bind to the frozen value and proceed.
# Done BEFORE the tee so a refusal never opens a log. C1_FROZEN_MIN_FREE_FILE relocates the
# records file (tests/records only); the exact-VALUE cross-check is the real protection.
FROZEN_FILE="${C1_FROZEN_MIN_FREE_FILE:-${EXPDIR}/exp03n_frozen_min_free.txt}"   # EXP03 DIFF 5/19: exp-09's frozen value is A6000-derived (handoff trap 3); THIS arm's L40 value is derived by exp03n_probe.sh and frozen here. Absent file => REFUSE.
[ -f "$FROZEN_FILE" ] || { echo "REFUSING TO LAUNCH: frozen-records file '${FROZEN_FILE}' is absent - exp03n_probe.sh (plan §4.4) has not FROZEN this arm's free-VRAM threshold yet."; exit 3; }   # EXP03 DIFF 16/19: refusal message names this arm's probe
FROZEN_RAW="$(cat "$FROZEN_FILE")"
FROZEN="$(printf '%s' "$FROZEN_RAW" | tr -d '[:space:]')"
case "$FROZEN" in
  ''|*[!0-9]*) echo "REFUSING TO LAUNCH: frozen value '${FROZEN_RAW}' in ${FROZEN_FILE} is not a plain integer."; exit 3;;
esac
MIN_FREE_MB="${MIN_FREE_MB:-$FROZEN}"
[ "$MIN_FREE_MB" = "$FROZEN" ] || { echo "REFUSING TO LAUNCH: MIN_FREE_MB=${MIN_FREE_MB} does not match the FROZEN C1 threshold ${FROZEN} (arbitrary override rejected - the launcher binds the exact derived value)."; exit 3; }

exec > >(tee -a "$LOG") 2>&1
echo "=== exp03n maxpool+MLP DDP+SyncBN - ${TS} - $(git rev-parse --short HEAD 2>/dev/null) - ${MB}x2x${ACC} eff64 seed42 -> 67500 - logger=${LOGGER} - job ${SLURM_JOB_ID:-nojob} ==="   # EXP03 DIFF 6/19: arm banner

# MIN_FREE_MB is bound to the FROZEN threshold above (= this arm's measured L40 peak x 1.15);   # EXP03 DIFF 18/19: threshold provenance
# it is NEVER exp-09's A6000 number nor B-F's inherited 21,900 MiB, and never an arbitrary override.   # EXP03 DIFF 19/19: threshold provenance

# --- per-GPU FREE-VRAM gate (fail-CLOSED on query errors; mirrors bf_scratch_launch.sh)
for G in 0 1; do
  FREE="$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
  rc_q=$?
  [ "$rc_q" -eq 0 ] && [ -n "$FREE" ] || { echo "nvidia-smi free-mem query failed on GPU ${G} (rc=${rc_q}) - refusing to launch blind"; exit 2; }
  [ "$FREE" -ge "$MIN_FREE_MB" ] || { echo "GPU ${G} free ${FREE} MiB < required ${MIN_FREE_MB} MiB - refusing to launch"; exit 2; }
done
echo "--- co-tenancy disclosure: compute apps at launch ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

# --- fail-closed wandb identity gate (only when wandb is requested) ---
if [ "$LOGGER" = "wandb" ]; then
  [ -r "$HOME/.netrc" ] || { echo "wandb identity: ~/.netrc is absent/unreadable - run 'wandb login' or use LOGGER=none - abort"; exit 2; }   # EXP03 DIFF 7/19: on this cluster wandb auth lives in ~/.netrc (the bashrc WANDB_API_KEY grep is dropped: it silently no-ops here and would let an unauthenticated run start)
  python - <<'PY'
import sys
try:
    import wandb
    email = wandb.Api().viewer.email
except Exception as e:
    print("wandb identity check FAILED:", e); sys.exit(1)
print("wandb identity:", email)
sys.exit(0 if email == "yh4742@princeton.edu" else 2)
PY
  gate=$?
  if [ "$gate" -ne 0 ]; then echo "wandb identity != yh4742@princeton.edu (exit ${gate}) - set the right WANDB_API_KEY or use LOGGER=none - abort"; exit 2; fi
fi

# --- pre-launch pin gate: custom-class + gauge + official-weight + config-delta
# (fail-closed, as B-F). Offline so the gate's own model construction cannot
# contact the Hub and mutate the cache it just validated.
EXPECT_PACKAGE_SHA="${EXPECT_PACKAGE_SHA:-}"; EXPECT_EXP09_SHA="${EXPECT_EXP09_SHA:-}"; [ -n "$EXPECT_PACKAGE_SHA" ] && [ -n "$EXPECT_EXP09_SHA" ] || { echo "REFUSING TO LAUNCH: both EXPECT_PACKAGE_SHA (cylindrical-dinov3 HEAD) and EXPECT_EXP09_SHA (worktree HEAD) must be set - no defaults, no fallback set"; exit 3; }   # EXP03 DIFF 8/19: BOTH pins REQUIRED
HF_HUB_OFFLINE=1 python "${EXPDIR}/assert_arm_configs_exp03n.py" --expect-package-sha "$EXPECT_PACKAGE_SHA" --expect-exp09-sha "$EXPECT_EXP09_SHA" || { echo "GATE FAILED - abort"; exit 1; }   # EXP03 DIFF 9/19: exp03n gate (config contract + live arm wiring + bitwise common-init)

echo "--- env manifest ---"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import cylindrical_dinov3 as c; print('cylindrical_dinov3', c.__version__, c.__file__)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader
pip freeze 2>/dev/null | sha256sum | awk '{print "pip-freeze sha256:", $1}'
echo "sync_batchnorm: true | strategy: ddp_find_unused_parameters_true | rung: ${MB}x2x${ACC} | MIN_FREE_MB=${MIN_FREE_MB}"

# --- optional RESUME passthrough (integrative-review finding 4; train.py:230 forwards
# --ckpt-path to trainer.fit(ckpt_path=...)). Set CKPT_PATH=<file> to resume the reviewed
# exact command. DISCLOSURE (plan §3): a Lightning resume is a FRESH stochastic continuation
# - RNG and dataloader position are NOT restored - so it is disclosed, never a bit-exact restart.
RESUME_ARGS=()
if [ -n "$CKPT_PATH" ]; then
  [ -f "$CKPT_PATH" ] || { echo "CKPT_PATH='${CKPT_PATH}' not found - abort"; exit 3; }
  RESUME_ARGS=(--ckpt-path "$CKPT_PATH")
  echo "--- RESUME from ${CKPT_PATH} (fresh stochastic continuation, plan §3) ---"
fi

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model-config "${EXPDIR}/FLAC_AR_exp03n.json" `# EXP03 DIFF 10/19: the exp03n model config = exp-09 + cond_pool/cond_mlp_hidden in BOTH ViT blocks (the ONLY model delta). The backtick-comment expands to nothing and adds no argument (verified).` \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 67500 --batch-size "$MB" --accum-batches "$ACC" --num-workers 6 --seed 42 \
  --num-gpus 2 --strategy ddp_find_unused_parameters_true --sync-batchnorm true \
  --logger "$LOGGER" --checkpoint-every 2500 \
  --name FLAC_exp03n_maxpoolmlp --experiment-name exp03n_maxpoolmlp `# EXP03 DIFF 11/19: arm run names` \
  --save-dir /n/fs/gatrdp/outputs/exp03n_maxpoolmlp `# EXP03 DIFF 12/19: save-dir on the cluster scratch (outside both repos, exp_02 convention)` \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
rc=$?
echo "=== exp03n maxpool+MLP exited rc=${rc} at $(date '+%Y-%m-%d %H:%M:%S') ==="   # EXP03 DIFF 13/19: arm banner
exit $rc
