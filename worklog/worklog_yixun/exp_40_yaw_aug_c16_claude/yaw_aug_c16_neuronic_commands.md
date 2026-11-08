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
# 1. <= 48 h before FULL (CPU/NAS only): the full AcousticRooms content re-hash (~40 min) and
#    the real-NAS primitive probe (O_EXCL, NOREPLACE rename, dir fsync, manifest; seconds)
/home/yixunhu/miniconda3/envs/flac/bin/python $K/exp40_kit.py ar-verify \
  --ar-root /media/diskstation/yixunhu/FLAC/AcousticRooms --manifest $K/exp40_ar_manifest.json \
  --out $K/receipts/ar_verify_a6000.json
/home/yixunhu/miniconda3/envs/flac/bin/python $K/exp40_kit.py nas-probe --out $K/receipts/nas_capability_a6000.json
# 2. the production-rung smoke on BOTH cards (30 steps, no checkpoint; writes the receipt FULL needs)
GPU_RESERVATION_OK=1 MODE=DDP_SMOKE PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 bash $K/exp40_train_local.sh
# 3. the 40k run (≈ 22 h exclusive); every 2,500-step checkpoint is NAS-acknowledged right after its
#    save (a bounded, killable copy); Ctrl-C / SIGTERM to the launcher terminates only its own
#    training group and releases its locks (exit 130)
GPU_RESERVATION_OK=1 MODE=FULL PLATFORM=a6000 RUNG=2x32 RUN_ID=exp40_C16_a6000_2x32 \
  nohup bash $K/exp40_train_local.sh > /dev/null 2>&1 &
# 4. after the launcher's PASS it has written the stable run-result manifest
#    $K/receipts/run_result_exp40_C16_a6000_2x32.json (names + sha256 of the check record, the
#    DDP_SMOKE reference, the FULL evidence, the audit; the training-time checkpoint dirpath).
#    Admission (CPU, ~2 min) derives every identity field from those validated records,
#    recomputes both ranks' 40k-step offset streams and re-hashes all 16 NAS checkpoints; the only
#    caller input is a free-text note. It refuses to overwrite an existing admission.
/home/yixunhu/miniconda3/envs/flac/bin/python $K/exp40_kit.py admit \
  --run-result $K/receipts/run_result_exp40_C16_a6000_2x32.json --receipts-dir $K/receipts \
  --note "exp_40 C16 FULL, A6000 2x32" --out $K/exp40_C16_admission.json
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

One-time prerequisites. Claims, submission states, the submission flock and the claim-probe receipt
live in ONE control directory shared by every checkout, `EXP40_CONTROL` (default
`/n/fs/gatrdp/exp40_control`; it must be on the same shared filesystem the jobs see):

```bash
# the dataset must reproduce the committed manifest (CPU node; ~1 h); <= 48 h before each submission
$PY $K/exp40_kit.py ar-verify --ar-root "$(readlink -f AcousticRooms)" --manifest $K/exp40_ar_manifest.json \
  --out $K/receipts/ar_verify_neuronic.json
# FULL only (valid 14 days): two tasks on two nodes, 20 barrier-fenced rounds racing O_EXCL inside
# $EXP40_CONTROL/claims; the verdict is written atomically to $EXP40_CONTROL/receipts/claim_probe.json
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
most conservative partition MaxTime line (every `sinfo` line is parsed and recorded in the submission
state). Under one flock in the control directory it reconciles every earlier execution of the same
RUN_ID against squeue + sacct and the durable claim owner (a duplicate job refused by the claim is
recorded, not counted as a failure), records the transition (live / terminal), and refuses while any is
live or ambiguous (a truncated state file is ambiguous); a failed execution needs a fresh identity: a
new PILOT stamp, or `RUN_ID=…_r1` for FULL. Only one job per execution can pass the claim. The job's
GPU gate inspects the devices Slurm allocated (by UUID, under the job's `CUDA_VISIBLE_DEVICES`). A FULL
job ends `LOCAL_COMPLETE_PENDING_PUBLICATION` (never PASS) and writes
`$K/receipts/run_result_exp40_C16_neuronic_<rung>.json`; PASS is the admission below.

**Copying the 40k (and all cadence) checkpoints to this box, publishing, admitting** — ON THIS BOX after
the FULL job ended `LOCAL_COMPLETE_PENDING_PUBLICATION`:

```bash
cd ~/codespace/exp-40-yaw-aug-c16; K=worklog/worklog_yixun/exp_40_yaw_aug_c16_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python; R=exp40_C16_neuronic_2x32     # or ..._8x8
S=~/exp40_neuronic_pull/$R; mkdir -p $S/receipts
rsync -a --partial <your-neuronic-login>:/n/fs/gatrdp/codespace/exp-40-yaw-aug-c16/outputs_FLAC/exp40_yaw_aug_c16/FLAC_exp40_C16/$R/ $S/
# the run result names (and hashes) exactly the four records admission needs
rsync -a <your-neuronic-login>:/n/fs/gatrdp/codespace/exp-40-yaw-aug-c16/$K/receipts/run_result_$R.json $S/receipts/
for f in $($PY -c "import json,sys; print(' '.join(v['name'] for v in json.load(open(sys.argv[1]))['files'].values()))" $S/receipts/run_result_$R.json); do
  rsync -a <your-neuronic-login>:/n/fs/gatrdp/codespace/exp-40-yaw-aug-c16/$K/receipts/$f $S/receipts/
done
$PY $K/exp40_kit.py nas-probe --out $K/receipts/nas_capability_a6000.json        # <= 48 h old
$PY $K/exp40_kit.py publish-cadence --src $S/checkpoints --run-id $R \
  --pulled-manifest $S/CHECKPOINTS.partial.sha256 --nas-receipt $K/receipts/nas_capability_a6000.json
# admission re-hashes all 16 NAS files and checks the 40k state against the training-time (neuronic)
# dirpath recorded in the run result -- no path to type
$PY $K/exp40_kit.py admit --run-result $S/receipts/run_result_$R.json --receipts-dir $S/receipts \
  --note "exp_40 C16 FULL, neuronic $R" --out $K/exp40_C16_admission.json
```

---

## C. Evaluation (this box, after admission) — see `yaw_aug_c16_eval_commands.md`
