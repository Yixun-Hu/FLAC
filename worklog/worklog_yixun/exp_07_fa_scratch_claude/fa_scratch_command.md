# Commands — exp_07_fa_scratch

Every command lands here at launch time (SOP). Training/eval launch templates will be appended when runs are commissioned.

## Config-identity audit round (2026-07-10 → 11; code `a3e8cf5`-era worktree, pre-commit)

```bash
# ckpt probe v1 (superseded)  → fa_scratch_2026-07-10_23:35:28_ckpt_probe.log
python worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:35:28_ckpt_probe.log

# ⚠ PROVENANCE DEVIATION (post-hoc record; flagged by the gpt-5.6-sol review, Blocking 1):
# the "embedded model_config vs repo" diff appended to the v1 log was produced by an
# INLINE python heredoc (not a checked-in script) run as:
#   python - <<'PY' ... torch.load FLAC.ckpt; flatten; diff vs FLAC_AR.json ... PY 2>&1 | tee -a <v1 log>
# Deviation from the universal-review + command-at-launch rules. Remediation: the diff
# logic now lives IN probe_released_ckpt.py (v2) and the v2 log supersedes the v1 log.

# arm configs (BV byte-copy; BF +2 training keys) + JSON diffs
cp src/configs/model_configs/FLAC/AR/FLAC_AR.json worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json
python - <<'PY'   # (inline, generative one-liner recorded verbatim; output committed as FLAC_AR_BF.json)
import json, collections
cfg = json.load(open("src/configs/model_configs/FLAC/AR/FLAC_AR.json"), object_pairs_hook=collections.OrderedDict)
cfg["training"]["cond_method"] = "fa_invariant"
cfg["training"]["frame_avg_angles"] = [0.0, 90.0, 180.0, 270.0]
json.dump(cfg, open("worklog/exp_07_fa_scratch_claude/FLAC_AR_BF.json", "w"), indent=4)
PY
diff <(python3 -m json.tool worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json) <(python3 -m json.tool src/configs/model_configs/FLAC/AR/FLAC_AR.json)
diff <(python3 -m json.tool worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json) <(python3 -m json.tool worklog/exp_07_fa_scratch_claude/FLAC_AR_BF.json)

# arm asserts v1 (superseded) → fa_scratch_2026-07-10_23:43:26_arm_asserts.log
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:43:26_arm_asserts.log

# ckpt probe v2 (canonical: all counter phases + in-file config diff + DINOv3 pin)
python worklog/exp_07_fa_scratch_claude/probe_released_ckpt.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-10_23:59:06_ckpt_probe_v2.log

# arm asserts v2 (canonical: factory wiring via configure_optimizers + seeded init-identity)
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-11_00:00:17_arm_asserts_v2.log
# (an intermediate v2 run at 23:59:25 caught the InverseLR step-0 warmup lr 5e-7 — assert
#  corrected to the closed-form expectation; that log retained as the red→green record)

# BatchNorm rebuttal evidence (Medium 3): 20 BatchNorm2d modules under context_audio.net.cnn.*
python - <<'PY'
import sys, os; sys.path.insert(0, os.getcwd())
import json, torch
from src.models.factory import create_model_from_config
m = create_model_from_config(json.load(open("worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json")))
print(len([n for n,mod in m.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]))
PY

# arm asserts v3 (v2 + fail-closed DINOv3 pin gate assert_vit_pin(); superseded by v4)
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-11_00:11:31_arm_asserts_v3.log

# arm asserts v4 (CANONICAL: pin gate env-resolved via huggingface_hub.constants.HF_HUB_CACHE
# + explicit raises surviving `python -O` — reverify2 fixes) with green + 2 red fail-closed tests:
python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py 2>&1 | tee -a <v4 log>            # green, exit 0
HF_HUB_CACHE=$(mktemp -d) python worklog/exp_07_fa_scratch_claude/assert_arm_configs.py         # red 1: empty cache -> RuntimeError, exit 1
HF_HUB_CACHE=$(mktemp -d) python -O worklog/exp_07_fa_scratch_claude/assert_arm_configs.py      # red 2: -O still raises, exit 1
# -> worklog/exp_07_fa_scratch_claude/fa_scratch_2026-07-11_00:16:07_arm_asserts_v4.log

## M0 fit/throughput ladder (GPU 1; LAUNCHED 2026-07-11, code e85ebde) — driver run verbatim:

```bash
# pre-launch gate (fail-closed ViT pin + arm identity), then ladder: B-F tries 64x1 -> 32x2 -> 16x4,
# first fit wins; B-V then runs the SAME pair (fit confirm + throughput anchor). 15 opt steps each,
# EMA on (config), HF_HUB_OFFLINE=1, workers 6, seed 42, logger none, save-dir scratch.
# Per attempt: background nvidia-smi VRAM sampler (5s) -> peak; throughput parsed from the PL bar.
# Full driver text: see the M0 section of fa_scratch_2026-07-11_<launch>_m0.log header (echoed at start).
# Acceptance: exit 0 + reached max_steps=15; CUDA OOM -> next rung.
```

## M0 EXTENDED ladder (GPU 1; LAUNCHED 2026-07-11 ~14:2x, post-amendment) — same driver shape,
## rungs 8x8 -> 4x16, VRAM sampler at 1 s; then B-V confirm at winner. Log: fa_scratch_*_m0ext.log

## B-V TRAINING LAUNCH (GPU 1; LAUNCHED 2026-07-11 ~14:5x, code 70dea5a) — pre-launch pin gate, then:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python train.py \
  --model-config worklog/exp_07_fa_scratch_claude/FLAC_AR_BV.json \
  --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
  --pretransform-ckpt-path weights/FLAC/VAE.safetensors \
  --max-steps 67500 --batch-size 8 --accum-batches 8 --num-workers 6 --seed 42 \
  --logger none --checkpoint-every 2500 --name FLAC_exp07_BV --experiment-name exp07_BV \
  --save-dir outputs_FLAC/exp07_BV
# env records (manifest): matmul precision + TF32 flags + pip freeze echoed into the launch log header
# screens: external eval_FLAC.py at 10k-step ckpts (K=8 eval-seed 42 full split bf16; EMA default path
#          + online via use_ema=false eval-config copy, committed before the first screen)
```

## B-V 10k-step SCREENS (GPU 1 co-located with B-V; first screen LAUNCHED 2026-07-12, code ecb8352; NEW worklog_yixun paths)

```bash
# per screen step S in {10000, 20000, ...}: EMA (default config) then ONLINE (use_ema=false copy),
# K=8, eval-seed 42, full unseen split, bf16 cond-autocast:
CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
  --model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BV.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --ckpt-path "outputs_FLAC/exp07_BV/epoch=2-step=10000.ckpt" \
  --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name exp07_BV_screen_S10000_ema
# then identically with --model-config .../FLAC_AR_BV_online_eval.json --eval-name ..._online
```

## GATE BLOCK (GPU 1; LAUNCHED 2026-07-14 ~21:4x post-B-V; code ecb8352) — 15 sequential evals:

```bash
# (1) gate evals: final ckpt outputs_FLAC/exp07_BV/epoch=14-step=67500.ckpt, EMA (FLAC_AR_BV.json),
#     K=8 (acousticroom_unseeneval.json) seeds 42..46 and K=1 (acousticroom_unseeneval_1.json) seeds 42..46
#     -> --eval-name exp07_BV_gate_K{8,1}_seed{S}, bf16, full unseen split
# (2) 291k corroborating screen (context-only): THEIR ckpt .../outputs_291k_scratch_vanilla/epoch=14-step=67500.ckpt
#     under OUR protocol: FLAC_AR_BV.json, K=8, seed 42 -> exp07_291k_corrob_K8_s42
# (3) selection-curve extras (EMA, K=8, seed 42): ckpts step in {27500, 32500, 62500, 65000}
#     -> exp07_BV_selcurve_S{step}  (67500 point = gate K8 seed42; 30k/40k/50k/60k = screen series)
CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py --model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BV.json \
  --dataset-config src/configs/dataset_configs/AR/eval/<per-K> --ckpt-path <per-run> \
  --cond-autocast bf16 --seed <S> --steps 1 --cfg-scale 1.0 --eval-name <per-run>
```

# M0 probe template (superseded by the launched drivers above):
#   for MB_ACC in "64 1" "32 2" "16 4": try B-F first (constrained arm), then pin the pair for BOTH arms
#   HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python train.py --model-config worklog/exp_07_fa_scratch_claude/FLAC_AR_BF.json \
#     --dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json \
#     --pretransform-ckpt-path weights/FLAC/VAE.safetensors --max-steps 15 --batch-size $MB --accum-batches $ACC \
#     --num-workers 6 --seed 42 --precision bf16-mixed --logger none --save-dir $SCRATCH/m0_probe ...
#   acceptance: >=10 steps, finite loss, no OOM; record peak VRAM + samples/s (EMA on)

## B-V EXTEND (GPU 1; LAUNCHED 2026-07-16 00:47, code c40908c) — resume 67.5k -> 100k adaptive:

```bash
LOGGER=none bash worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh 100000
# -> fa_scratch_2026-07-16_00-47-25_BVextend_train.log ; ckpts outputs_FLAC/exp07_BVextend/ every 2500
# logger none: wandb held until the yh4742@princeton.edu key replaces the yixunhu21 one (fail-closed gate in script)
```

## B-V EXTEND SCREENS (GPU 1 co-located; per landed 10k ckpt S in {70000,80000,90000,100000}):

```bash
# EMA (default config) then ONLINE (use_ema=false copy), K=8, eval-seed 42, full unseen split, bf16:
CUDA_VISIBLE_DEVICES=1 python eval_FLAC.py \
  --model-config worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BV.json \
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json \
  --ckpt-path "outputs_FLAC/exp07_BVextend/epoch=<E>-step=<S>.ckpt" \
  --cond-autocast bf16 --seed 42 --steps 1 --cfg-scale 1.0 --eval-name exp07_BVext_screen_S<S>_ema
# then identically with --model-config .../FLAC_AR_BV_online_eval.json --eval-name ..._online
# tee -> fa_scratch_<ts>_BVext_screen_S<S>.log
```

## B-F FROM-SCRATCH (GPU 1; PRE-STAGED 2026-07-16 ~05:00, Yixun slot-go: extend → B-F → P1; launches when extend completes ~Jul 17 16:00):

```bash
# (SUPERSEDED 2026-07-16 pm: single-GPU 8x8 plan replaced by Yixun's DDP+SyncBN mandate below)

## M1 DDP+SyncBN fit probe (both GPUs; runs when aug291k frees GPU 0, ~Jul 18 02:00):
bash worklog/worklog_yixun/exp_07_fa_scratch_claude/m1_ddp_fit_probe.sh
# single BN-compliant rung 32/GPU x 2 x accum 1 (accumulation never feeds BN stats ->
# 16x2x2 / 8x2x4 would violate BN=64; review-eliminated). 15 steps, SyncBN on, timeout -k,
# dual VRAM samplers. Fit -> REPORT RUNG TO YIXUN AND WAIT FOR GO. No-fit -> STOP, options to Yixun.

## B-F FROM-SCRATCH DDP+SyncBN (LAUNCH-GATED on Yixun's post-probe go):
LOGGER=wandb MB=32 ACC=1 bash worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh
# 32/GPU x 2 x 1 = eff 64, SyncBN batch 64 = paper (deliberate deviation from release CODE:
# release got BN-64 via micro-64 single-H100, no SyncBN). --num-gpus 2
# --strategy ddp_find_unused_parameters_true --sync-batchnorm true; seed 42, 67500 steps,
# ckpt/2500; otherwise byte-identical flags to the B-V manifest. Fail-closed: both-GPU-free
# guard, MB*2==64 BN invariant, wandb identity gate (yh4742@princeton.edu, key self-extracted),
# offline pin gate, train.py refuses sync_batchnorm below 2 GPUs.
# wandb project FLAC_exp07_BF run exp07_BF; ckpts nest under outputs_FLAC/exp07_BF/<proj>/<run>/checkpoints/.
# Screens per 10k ckpt: bash .../bf_screen.sh <step>  (recursive ckpt find)
```

# consolidated review (first gpt-5.6-sol use) + focused re-verify + terse fix-verify
~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  --output-last-message worklog/exp_07_fa_scratch_claude/fa_scratch_codex_code_audit_probes_review.md "<context-briefed prompt>" < /dev/null
~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  --output-last-message worklog/exp_07_fa_scratch_claude/fa_scratch_codex_code_audit_probes_reverify.md "<fix-list prompt>" < /dev/null
~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  --output-last-message worklog/exp_07_fa_scratch_claude/fa_scratch_codex_code_audit_probes_reverify2.md "<residual fix-list prompt>" < /dev/null
```
