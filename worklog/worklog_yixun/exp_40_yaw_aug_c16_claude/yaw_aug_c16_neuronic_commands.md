# exp_40 yaw_aug_c16 — launch command sheet (for Yixun)

Plan: `plan_yaw_aug_c16.md` (r3 FINAL; §0 decisions D1–D6, §14 the binding fixes). The platform is
**your decision (D1)**: (a) 2×A6000 here — RECOMMENDED, the figure's YAW arm (`ac1f2603…`, exp_17) was
trained exactly this way; (b) neuronic 2×L40 × 32 — same rung and draw keys, different GPU; (c) neuronic
8×L40 × 8 — exp_15's recipe, not comparable with the figure arm. Every command below needs the pushed
launch commit `EXPECT_SHA` (the final reviewed commit of branch `exp-40-yaw-aug-c16`; this sheet is
updated with it at hand-off).

---

## A. 2×A6000 on this box (D1 a, primary)

Run from the exp_40 worktree `~/codespace/exp-40-yaw-aug-c16` at `EXPECT_SHA` (tracked tree clean). The
launcher refuses unless both cards are free (`GPU_POLICY=idle`, uuid-mapped) and takes the cooperative
per-card locks `~/codespace/claude_gpu_yield/GPU_LOCK_<uuid>`; sequence the cards through that handshake
first (REQUEST → PAUSED from the other sessions), then set `GPU_RESERVATION_OK=1`.

```bash
cd ~/codespace/exp-40-yaw-aug-c16
export EXPECT_SHA=<40-hex launch commit>
K=worklog/worklog_yixun/exp_40_yaw_aug_c16_claude
# 0. every gate, nothing spawned
MODE=FULL PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 DRY_RUN=1 bash $K/exp40_train_local.sh
# 1. full AcousticRooms content re-hash, <= 48 h before FULL (CPU/NAS only, ~40 min)
/home/yixunhu/miniconda3/envs/flac/bin/python $K/exp40_kit.py ar-verify \
  --ar-root /media/diskstation/yixunhu/FLAC/AcousticRooms --manifest $K/exp40_ar_manifest.json \
  --out $K/receipts/ar_verify_a6000.json
# 2. the production-rung smoke on BOTH cards (30 steps, no checkpoint; writes the receipt FULL needs)
GPU_RESERVATION_OK=1 MODE=DDP_SMOKE PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 bash $K/exp40_train_local.sh
# 3. the 40k run (≈ 22 h exclusive); every 2,500-step checkpoint is NAS-acknowledged right after its save
GPU_RESERVATION_OK=1 MODE=FULL PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 \
  nohup bash $K/exp40_train_local.sh > /dev/null 2>&1 &
# 4. after a PASS (evidence + audit): admission (CPU)
/home/yixunhu/miniconda3/envs/flac/bin/python $K/exp40_kit.py admit \
  --nas-run-dir /media/diskstation/yixunhu/FLAC/checkpoints/exp40_yaw_aug_c16/exp40_C16_a6000_2x32 \
  --expected-dirpath $PWD/outputs_FLAC/exp40_yaw_aug_c16/FLAC_exp40_C16/exp40_C16_a6000_2x32/checkpoints \
  --evidence $K/receipts/full_evidence_exp40_C16_a6000_2x32.json --audit $K/yaw_aug_c16_<ts>_FULL_audit.json \
  --meta-json '{"run_id": "exp40_C16_a6000_2x32", "platform": "a6000", "rung": "2x32", "train_head": "'$EXPECT_SHA'"}' \
  --out $K/exp40_C16_admission.json
```

Crash (D3): a crashed FULL is NOT resumed; rerun from step 0 with `RUN_ID=exp40_C16_a6000_2x32_r1` (a new
namespace). The only resume command that exists is manual and makes a *sensitivity* arm, never the
primary: `train.py <the FULL argv with --experiment-name exp40_C16_a6000_2x32_resumed<N>> --ckpt-path <step-N ckpt>`.

---

## B. neuronic (D1 b/c, non-primary; you submit)

**Getting the branch onto neuronic** (repo `/n/fs/gatrdp/codespace/FLAC`):

```bash
cd /n/fs/gatrdp/codespace/FLAC
git fetch origin exp-40-yaw-aug-c16
git worktree add --detach /n/fs/gatrdp/codespace/exp-40-yaw-aug-c16 <EXPECT_SHA>
cd /n/fs/gatrdp/codespace/exp-40-yaw-aug-c16
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/AcousticRooms)" AcousticRooms
ln -s "$(readlink -f /n/fs/gatrdp/codespace/FLAC/weights/FLAC/VAE.safetensors)" weights/FLAC/VAE.safetensors
K=worklog/worklog_yixun/exp_40_yaw_aug_c16_claude; PY=/n/fs/gatrdp/envs/flac/bin/python
```

One-time prerequisites (each writes a receipt under `$K/receipts/` that the submitter checks):

```bash
# the dataset must reproduce the committed manifest (CPU node; ~1 h)
$PY $K/exp40_kit.py ar-verify --ar-root "$(readlink -f AcousticRooms)" --manifest $K/exp40_ar_manifest.json \
  --out $K/receipts/ar_verify_neuronic.json
# FULL only: two nodes racing O_EXCL on the shared mount must give exactly one winner per round
sbatch --output=$PWD/$K/slurm_claim_probe_%j.out $K/exp40_claim_probe.sbatch
```

The sealed HF hub is built by the submitter from `/n/fs/gatrdp/hf_cache/hub` — the pinned DINOv3 snapshot
`114c1379950215c8b35dfcd4e90a5c251dde0d32` must be there (config.json `9481247b…`, model.safetensors `4610ad75…`).

```bash
# PILOT: 300 steps on the target rung (timing + the receipt FULL needs). RUNG=2x32 (recommended) or 8x8
DRY_RUN=1 MODE=PILOT RUNG=2x32 RUN_ID=exp40_C16_neuronic_2x32 EXPECT_SHA=<sha> bash $K/exp40_submit_neuronic.sh
MODE=PILOT RUNG=2x32 RUN_ID=exp40_C16_neuronic_2x32 EXPECT_SHA=<sha> bash $K/exp40_submit_neuronic.sh
# FULL: needs YOUR advance NAS relaxation (plan D4): the 16 checkpoints stay on neuronic until the run ends
EXP40_NAS_RELAXATION=post-run-publish MODE=FULL RUNG=2x32 RUN_ID=exp40_C16_neuronic_2x32 EXPECT_SHA=<sha> \
  bash $K/exp40_submit_neuronic.sh
```

The submitter derives `--time` from the PILOT's measured rate (×1.15 + 45 min) and refuses above the
partition MaxTime; it refuses a new submission while an earlier execution of the same RUN_ID is live or
ambiguous (no `NEW_INTENT`; a failed execution needs a fresh identity: a new PILOT stamp, or
`RUN_ID=…_r1` for FULL). Only one job per execution can pass the claim.

**Copying the 40k (and all cadence) checkpoints to this box** — run ON THIS BOX after the FULL job PASSED:

```bash
cd ~/codespace/exp-40-yaw-aug-c16; K=worklog/worklog_yixun/exp_40_yaw_aug_c16_claude
S=~/exp40_neuronic_pull; mkdir -p $S
rsync -a --partial <your-neuronic-login>:/n/fs/gatrdp/codespace/exp-40-yaw-aug-c16/outputs_FLAC/exp40_yaw_aug_c16/FLAC_exp40_C16/exp40_C16_neuronic_2x32/ $S/
rsync -a <your-neuronic-login>:/n/fs/gatrdp/codespace/exp-40-yaw-aug-c16/$K/receipts/full_evidence_exp40_C16_neuronic_2x32.json $K/receipts/
/home/yixunhu/miniconda3/envs/flac/bin/python $K/exp40_kit.py publish-cadence --src $S/checkpoints \
  --run-id exp40_C16_neuronic_2x32 --pulled-manifest $S/CHECKPOINTS.partial.sha256
# then `audit` (with --nas-dir) and `admit` exactly as in A.4, with the neuronic run id / dirpath
```

---

## C. Evaluation (this box, after admission) — see `yaw_aug_c16_eval_commands.md`
