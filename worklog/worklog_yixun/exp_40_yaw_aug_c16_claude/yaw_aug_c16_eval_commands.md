# exp_40 yaw_aug_c16 — evaluation plan and ready-to-run commands (this box)

Plan §5 + §14 H7/H8. Protocol = the exp_25 YAW-pin cells' argv (copied from the committed
`exp_25…/depth_canon_cells.json` rows; only `--model-config`, `--seed`, `--rotate-*`, `--eval-name`
change), EMA, full unseen split (6,337 items / 17 rooms), vanilla conditioning, bf16 conditioner autocast,
cfg 1.0, 1 step, batch 64, stream + per-scene records, scored-waveform dumps to
`/media/diskstation/yixunhu/FLAC/exp40_pred_waveforms/` (announcement 08), **SDPA** through the committed
no-flash launcher `overlays/no_flash/exp35_no_flash_launch.py` (the exp_25 YAW cells' backend).

## Cells (85 new; `exp40_cells.json`)

| Block | Arm | Angles (deg) | K | Seeds | Cells |
|---|---|---|---|---|---|
| E0 replays (hard stop) | YAW | T 0 (K1), C4 90 (K8), R (K8) | | 42 | 3 |
| T (θ = 0) | C16 | 0 | 1, 8 | 42–46 | 10 |
| C4 | C16 | 90, 180, 270 | 1, 8 | 42–46 | 30 |
| 45°-grid (ON the C16 grid) | C16 | 45, 135, 225, 315 | 1, 8 | 42 | 8 |
| off-grid (half steps) | C16 | 11.25, 101.25, 191.25, 281.25 | 1, 8 | 42 | 8 |
| random yaw (512 columns, rotate seed = eval seed) | C16 | random | 1, 8 | 42–46 | 10 |
| FILL (comparators missing at the off-grid angles) | YAW, P1 | 11.25, 101.25, 191.25, 281.25 | 1, 8 | 42 | 16 |

Reused (never re-run), pinned in `exp40_comparators.json`: exp_25 YAW / P1 T (s42–46), R (s42–46), C4 + 45
(s42); exp_36 YAW / P1 135 / 225 / 315 (s42) and YAW C4 s43–46 (all with dumps). The random-yaw R cells for
YAW and P1 already exist (exp_25), so none are added. Off-grid set: plan D5 (the brief's
{11.25, 33.75, 56.25, 101.25} would add 24 cells ≈ 4.9 GPU-h).

**GPU time:** 42 × 11.9 + 43 × 12.6 min ≈ **17.4 GPU-h ≈ 8.7 h on two exclusive A6000s** (≈ 17.4 h on
one; 1.6–5× longer next to a 100 %-utilisation co-tenant). E0 alone ≈ 37 min. NAS dumps ≈ 22 GB.

## Order (plan §14 H8) and commands

```bash
cd ~/codespace/exp-40-yaw-aug-c16; K=worklog/worklog_yixun/exp_40_yaw_aug_c16_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python
# 1. after exp_36 has published: pin the reused cells, commit (with the figure contract), push
$PY $K/exp40_eval.py register-comparators \
  --exp25-farm ~/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon \
  --exp36-farm ~/codespace/exp-36-run/outputs_FLAC/exp25_depth_canon
git add $K/exp40_comparators.json && git commit -m "exp_40: comparator registry" && git push
EVAL_SHA=$(git rev-parse HEAD)
# 2. the detached eval worktree at EVAL_SHA (the lab's run-worktree pattern) + its allowlisted assets
git worktree add --detach ~/codespace/exp-40-eval-${EVAL_SHA:0:12} $EVAL_SHA
W=~/codespace/exp-40-eval-${EVAL_SHA:0:12}
ln -s /media/diskstation/yixunhu/FLAC/AcousticRooms $W/AcousticRooms
ln -s ~/codespace/FLAC/weights/AGREE $W/weights/AGREE
ln -s ~/codespace/FLAC/weights/FLAC/VAE.ckpt $W/weights/FLAC/VAE.ckpt
mkdir -p /media/diskstation/yixunhu/FLAC/exp40_pred_waveforms
# 3. full AcousticRooms re-hash before freezing the suite (<= 48 h), then the E0 hard stop (one free card)
$PY $K/exp40_kit.py ar-verify --ar-root /media/diskstation/yixunhu/FLAC/AcousticRooms \
  --manifest $K/exp40_ar_manifest.json --out $K/receipts/ar_verify_eval.json
SEALED=$PWD/outputs_FLAC/exp40_yaw_aug_c16/hf_sealed
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree $W --eval-sha $EVAL_SHA --hf-sealed $SEALED \
  --blocks E0 --lanes 0 --dry-run            # prints the 3 argv, nothing runs
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree $W --eval-sha $EVAL_SHA --hf-sealed $SEALED \
  --blocks E0 --lanes 0                      # exit 0 + receipts/e0_<EVAL_SHA>_<closure>.json EQUAL, else STOP
# 4. after the C16 admission (exp40_kit.py admit): the hash-pinned runtime-input manifest
$PY -c "import hashlib,json,sys; p=sys.argv[1]; json.dump({'admission': {'path': p, 'sha256': hashlib.sha256(open(p,'rb').read()).hexdigest()}}, open(sys.argv[2],'w'))" \
  $PWD/$K/exp40_C16_admission.json $K/receipts/runtime_inputs_C16.json
# 5. every remaining cell (both cards when free; the driver waits for an idle card, never signals anything)
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree $W --eval-sha $EVAL_SHA --hf-sealed $SEALED \
  --runtime-inputs $K/receipts/runtime_inputs_C16.json --blocks T,C4,G45,OFF,R,FILL --lanes 0,1
# 6. tables + off-grid readout, then the exp_40 figure (validated against exp40_figure_contract.json)
$PY $K/exp40_eval.py collect --repo $PWD --tree $W
$PY $K/exp40_eval.py figure --repo $PWD --tree $W
```

Outputs: `yaw_aug_c16_cells_raw.csv`, `yaw_aug_c16_results_tables.md`, `yaw_aug_c16_figure.{png,pdf}` +
`yaw_aug_c16_figure_manifest.json` in this folder. The paper repo's three-model script is not edited here
(plan D6); `yaw_aug_c16_cells_raw.csv` is its import file. `model_comparison.md`: the C16 T rows publish
with the next neuronic regeneration in one results update (plan §5.4).
