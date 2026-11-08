# exp_40 yaw_aug_c16 — evaluation plan and ready-to-run commands (this box)

Plan §5 under the binding §14 (H7/H8), as fixed for Codex code review 1 (B7–B10, H4, H7–H12, M5).
Protocol = the exp_25 YAW-pin cells' argv (copied from the committed
`exp_25…/depth_canon_cells.json` rows, file sha `1073ef64…` pinned; only `--model-config`, `--seed`,
`--rotate-*`, `--eval-name` change), EMA, full unseen split (6,337 items / 17 rooms), vanilla conditioning,
bf16 conditioner autocast, cfg 1.0, 1 step, batch 64, stream + per-scene records, scored-waveform dumps to
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

Reused (never re-run), pinned in `exp40_comparators.json` (92 cells, each validated by its source
experiment's own `depth_canon_collect.verify_cell` at its pin, `d6ca307a` / `6be49f46`, with a full dump
re-hash, then by exp_40's strict checks; exp_36 cells also bind the replay verdict that admitted their
pin): exp_25 YAW / P1 T (s42–46), R (s42–46), C4 + 45 (s42); exp_36 YAW / P1 135 / 225 / 315 (s42) and
YAW C4 s43–46. Off-grid set: plan D5 (the brief's {11.25, 33.75, 56.25, 101.25} would add 24 cells
≈ 4.9 GPU-h).

**GPU time:** 42 × 11.9 + 43 × 12.6 min ≈ **17.4 GPU-h ≈ 8.7 h on two exclusive A6000s** (≈ 17.4 h on
one; 1.6–5× longer next to a 100 %-utilisation co-tenant). E0 alone ≈ 37 min. NAS dumps ≈ 22 GB.

## What every `run` checks (and refuses on, exit 2)

The eval worktree (HEAD = EVAL_SHA, detached, clean, only the three allowlisted untracked links, pinned
assets / configs, sealed HF hub, IR counts of the 17 unseen rooms) — before EVERY spawn; the runtime-input
manifest (exact names `ar_verify` [+ `admission` for C16], absolute paths OUTSIDE the eval worktree, every
file hashing to its pin); a 0–48 h full AcousticRooms re-hash receipt of the committed manifest over this
root + the 16-item eval-split content sample (`AR_EVAL_SAMPLE_DIGEST ce890a70…`) — before every cell batch;
the environment probe (pinned interpreter and versions, module origins inside the env, the eval tree's
`src.models.transformer`, the no-flash `--preflight` sentinel, offline HF resolution inside the sealed hub,
NVIDIA driver `590.48.01`) whose digest is bound into the E0 receipt and every cell receipt; the C16
admission through `exp40_kit.admission_reasons` (40k checkpoint re-hashed); the exp_40 card lock
`~/.exp40_cards.lock` for the whole run; per cell the cooperative `claude_gpu_yield/GPU_LOCK_<uuid>` of the
lane's card, re-checked (A6000, no compute process, ≥ 10 GiB free) under the lock immediately before the
spawn and held for the evaluator's lifetime. A signal terminates only the driver's own evaluator process
groups (TERM, 60 s, KILL) and releases its locks. Success = every selected cell ends with exactly one
receipt that re-verifies now. `--dry-run` runs every one of these gates and only skips the locks and the
spawns: any failing gate is the same `REFUSED:` line and exit 2 (code-r2 M2).

**Runtime identity (code-r2 B4).** Every cell receipt records its runtime inputs (exact `{path, sha256}`
entries) and their canonical digest, and its start time; ONE validator — at the spawn, in `run`,
`verify`, `collect`, `figure` and the offline table validator — recomputes that digest, re-hashes every
recorded input, requires the C16 admission to be the validated one and to predate the cell, and requires
the recorded AR re-hash receipt to have been a PASS of the committed manifest **0–48 h old at that cell's
start**. The E0 receipt's runtime-input / AR / dataset-config / time fields are re-derived from its three
cell receipts. Hence: AR receipts and runtime-input manifests are **never overwritten** (timestamped
names below) and stay in `$R` (inside this repo, outside the eval worktree) — they are committed with the
table evidence at results time.

## Order (plan §14 H8) and commands

```bash
cd ~/codespace/exp-40-yaw-aug-c16; K=worklog/worklog_yixun/exp_40_yaw_aug_c16_claude
PY=/home/yixunhu/miniconda3/envs/flac/bin/python; R=$PWD/$K/receipts; mkdir -p "$R"
SEALED=$PWD/outputs_FLAC/exp40_yaw_aug_c16/hf_sealed
# 0. one-time pins. The E0 reference is COMMITTED (exp40_e0_reference.json, sha 180a4534…, pinned in
#    exp40_eval.py); it was built with (refuses if the file exists):
#      $PY $K/exp40_eval.py build-e0-reference --out $K/exp40_e0_reference.json
# 1. the comparator registry (exp_36 has published: 30 YAW + 6 P1 cells at 6be49f46). ~19 GB NAS read
#    (the source validators re-hash every dump); then commit it with the figure contract and push
nice -n 15 ionice -c3 $PY $K/exp40_eval.py register-comparators --out $K/exp40_comparators.json \
  2>&1 | tee $K/yaw_aug_c16_$(date +%F_%H-%M-%S)_register_comparators.log
git add $K/exp40_comparators.json && git commit -m "exp_40: comparator registry" && git push
EVAL_SHA=$(git rev-parse HEAD)
# 2. the detached eval worktree at EVAL_SHA (the lab's run-worktree pattern) + its three allowlisted assets
W=~/codespace/exp-40-eval-${EVAL_SHA:0:12}
git worktree add --detach "$W" "$EVAL_SHA"
ln -s /media/diskstation/yixunhu/FLAC/AcousticRooms "$W/AcousticRooms"
ln -s ~/codespace/FLAC/weights/AGREE "$W/weights/AGREE"
ln -s ~/codespace/FLAC/weights/FLAC/VAE.ckpt "$W/weights/FLAC/VAE.ckpt"
mkdir -p /media/diskstation/yixunhu/FLAC/exp40_pred_waveforms          # the dump root is never auto-created
# 3. the sealed HF hub (built once from the local cache; `run --dry-run` below verifies it + resolution)
[ -d "$SEALED" ] || $PY $K/exp40_kit.py build-hf-sealed --src-hub ~/.cache/huggingface/hub --dest "$SEALED"
# 4. the full AcousticRooms re-hash (CPU, ~1 h, NAS), outside the eval worktree, to a NEW timestamped
#    receipt (never overwrite one: earlier cell receipts are bound to its bytes), and a NEW runtime-input
#    manifest pinning it. Every cell must START within 48 h of the receipt's time: before a batch that
#    could run past that, repeat step 4 (and step 6) and use the new manifest.
pin() { $PY -c "import hashlib,json,sys; print(json.dumps({'path': sys.argv[1], 'sha256': hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()}))" "$1"; }
AR="$R/ar_verify_eval_$(date +%F_%H-%M-%S).json"; [ -e "$AR" ] && echo "exists: $AR" && exit 1
$PY $K/exp40_kit.py ar-verify --ar-root /media/diskstation/yixunhu/FLAC/AcousticRooms \
  --manifest $K/exp40_ar_manifest.json --out "$AR"
RI_E0="$R/runtime_inputs_E0_$(date +%F_%H-%M-%S).json"
echo "{\"ar_verify\": $(pin "$AR")}" > "$RI_E0"
# 5. the E0 hard stop (one free card): dry run (every gate + the 3 argv; exit 2 on any refused gate;
#    nothing locked / spawned), then the run
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree "$W" --eval-sha $EVAL_SHA --hf-sealed "$SEALED" \
  --runtime-inputs "$RI_E0" --blocks E0 --lanes 0 --dry-run || { echo "E0 dry run REFUSED: stop"; exit 1; }
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree "$W" --eval-sha $EVAL_SHA --hf-sealed "$SEALED" \
  --runtime-inputs "$RI_E0" --blocks E0 --lanes 0 \
  2>&1 | tee $K/yaw_aug_c16_$(date +%F_%H-%M-%S)_eval_E0.log
#    exit 0 + receipts/e0_<EVAL_SHA>_<closure>.json with "verdict": "EQUAL" -- anything else: STOP, report
# 6. after the C16 admission (yaw_aug_c16_neuronic_commands.md / the launch sheet: exp40_kit.py admit
#    -> $K/exp40_C16_admission.json): a NEW runtime-input manifest with BOTH inputs; $AR must be a
#    receipt less than 48 h old for every cell of the batch (else repeat step 4 first)
RI_C16="$R/runtime_inputs_C16_$(date +%F_%H-%M-%S).json"
echo "{\"ar_verify\": $(pin "$AR"), \"admission\": $(pin "$PWD/$K/exp40_C16_admission.json")}" > "$RI_C16"
# 7. every remaining cell (both cards when free; waits for an idle card, never signals anything not its
#    own); a dry run first (exit 2 on any refused gate, incl. a missing / non-EQUAL E0 receipt)
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree "$W" --eval-sha $EVAL_SHA --hf-sealed "$SEALED" \
  --runtime-inputs "$RI_C16" --blocks T,C4,G45,OFF,R,FILL --lanes 0,1 --dry-run || { echo "REFUSED: stop"; exit 1; }
$PY $K/exp40_eval.py run --repo $PWD --eval-worktree "$W" --eval-sha $EVAL_SHA --hf-sealed "$SEALED" \
  --runtime-inputs "$RI_C16" --blocks T,C4,G45,OFF,R,FILL --lanes 0,1 \
  2>&1 | tee $K/yaw_aug_c16_$(date +%F_%H-%M-%S)_eval_cells.log
#    a cell whose spawn would start more than 48 h after $AR's time is refused ("runtime inputs refused at
#    the cell's start"): repeat steps 4 + 6 and re-run step 7 with the new manifest (verified cells skip)
# 8. re-verify every receipt with the same validators (non-zero on anything missing / invalid; each
#    receipt is checked against the runtime inputs IT recorded; the manifest here supplies the admission)
$PY $K/exp40_eval.py verify --repo $PWD --eval-worktree "$W" --eval-sha $EVAL_SHA --runtime-inputs "$RI_C16"
# 9. collect (CSV + every table + off-grid readout incl. P1's d + paper-import CSV + collect manifest),
#    then the figure from the frozen, re-validated CSV
$PY $K/exp40_eval.py collect --repo $PWD --tree "$W" --eval-sha $EVAL_SHA --admission $PWD/$K/exp40_C16_admission.json
$PY $K/exp40_eval.py figure  --repo $PWD --tree "$W" --eval-sha $EVAL_SHA --admission $PWD/$K/exp40_C16_admission.json
# 10. results time: the table evidence the offline row validator checks; force-add + commit it with every
#     file it names -- the ten T metrics / stream / receipt files, the admission, the E0 receipt, the three
#     E0 cell receipts and every AR re-hash receipt those receipts ran under (plan §5.4; code-r2 B4)
$PY $K/exp40_eval.py freeze-table-evidence --repo $PWD --tree "$W" --eval-sha $EVAL_SHA \
  --admission $PWD/$K/exp40_C16_admission.json --out $K/exp40_table_evidence.json
$PY -c "import json,sys; e=json.load(open(sys.argv[1])); print('\n'.join([e['admission']['path'], e['e0_receipt']['path']] + [v['path'] for v in e['e0_cells'].values()] + list(e['ar_receipts'].values()) + [f['path'] for c in e['cells'].values() for f in c.values()]))" \
  $K/exp40_table_evidence.json | xargs git add -f && git add $K/exp40_table_evidence.json
```

Outputs (this folder): `yaw_aug_c16_cells_raw.csv`, `yaw_aug_c16_results_tables.md` (θ = 0, C4, 45°-grid,
off-grid and random-yaw per-angle tables + the off-grid readout), `yaw_aug_c16_paper_import.csv` (the
column schema of the paper repo's `fig_inputs/appendix/*_raw.csv` archive inputs; fractional angles kept as
`11.25`, which that script's `int(angle_deg)` does not parse yet — the change is Yixun's, plan D6),
`yaw_aug_c16_collect_manifest.json`, `yaw_aug_c16_figure.{png,pdf}` + `yaw_aug_c16_figure_manifest.json`
(CSV sha, collect-manifest sha, contract sha, only the contract-selected cells). `model_comparison.md`:
the C16 T rows publish with the next neuronic regeneration in one results update (plan §5.4); the generator
requires exactly the five canonical files per K and the committed table evidence.
