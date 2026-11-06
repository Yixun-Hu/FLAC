# exp_36 yaw_c4_spread — plan r0 (2026-11-06, Planner: Claude Opus 5.5)

**Request (Yixun, 2026-11-06 14:35 + 14:40, relayed by the desktop session; verbatim in `yaw_c4_spread_yixun_query.md`):**
complete the Yaw-Augmented FLAC C4 spread for the paper's three-model C4 yaw figure
(`-Neurips-2026-workshop-Sound-Localization/scripts/plot_yaw_c4_three_models.py`): 24 cells = the exp_25 YAW@40k
checkpoint × θ ∈ {90, 180, 270} × K ∈ {1, 8} × eval seeds 43–46, with exactly the flags of the existing exp_25
YAW_C4 seed-42 cells, through the exp_35 no-flash backend; first re-run one archived seed-42 cell and require exact
equality with its JSON as a HARD STOP, then the 24; outputs next to the existing exp_25 YAW cells. Priority over
exp_35 phase (ii) and exp_22; cards via the `claude_gpu_yield/` handshake (REQUEST → PAUSED → run → RELEASE).
Review chain kept short by his order: this brief plan + ONE Codex plan review, then go.

## 1. Facts this plan rests on (all read-only checks, 2026-11-06)
- Checkpoint: `/media/diskstation/yixunhu/FLAC/checkpoints/ar_40k_endpoints/YAW/epoch=8-step=40000.ckpt`,
  sha256 `ac1f26034e4f341fe0c2cb4638e2eb473959d66ddd2fd95d184dc2fd4f264de7` (exp_25 manifest `arms.YAW`).
- The template rows exist in the H35 manifest (`5b065801`, `exp_25_depth_canon_claude/depth_canon_cells.json`,
  block YAW-pin, subblock C4, orders 92–98): e.g. `exp25_YAW_C4_S40000_K8_s42_th90` → eval argv
  `--model-config worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json --dataset-config
  src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json (K=1: acousticroom_unseeneval_1.json)
  --cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --device cuda
  --record-per-scene --record-stream --expected-stream-count 6337 --seed 42 --rotate-mode fixed --rotate-deg 90
  --eval-name exp25_YAW_C4_S40000_K8_s42_th90`, dump on.
- The archived seed-42 outputs (exp_25 campaign, source pin `d6ca307a`, SDPA — flash-attn was not installed then):
  `~/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/YAW/epoch=8-step=40000_metrics_1_1.0_<cell>_rot<θ>.json`
  (+ `.stream.json`, `.cell.json` receipt). The figure script reads exactly this directory and file pattern and checks
  `rotate_deg`, `seed`, `cond_method == vanilla`, `ckpt_sha256_loaded` prefix `ac1f26034e4f`.
- The replay reference (pinned now): `…_exp25_YAW_C4_S40000_K8_s42_th90_rot90.json` sha256 `fc8cb212a9a5b852d1ad3bc31e01ff17c1b0b93b79141d92d0c016751eb11130` (5,250 B), `.stream.json` `e671270fc3cd61f5a47dd68f6f41707db68d6e62b4f2750ae52e9e7f9b7085a8` (3,691,911 B), `.cell.json` receipt `ee7d624a0c5a5df589fa7e47690aca85e2d6297007a466ad69820bf3c7bf9177` (4,222 B); exp-25 worktree HEAD `5b7e663b`.
- In the H35 code the no-flash overlay applies only to `EXP35_BLOCKS` (`depth_canon_collect.py:45`); the YAW-pin block
  is NOT in it, so re-running a YAW-pin row at H35 would use flash-attn and could not reproduce exp_25.
- Measured exclusive-card cell times for YAW (exp_35 cell-time table): K1 median 11.9 min, K8 median 12.6 min.

## 2. Design
- **Separate worktree** `~/codespace/exp-36-yaw-c4-spread`, branch `exp-36-yaw-c4-spread` cut from H35 `5b065801`; the
  exp_35 run worktree, its manifest and its campaign digest stay untouched (phase (ii) remains bound to H35).
- **One code round (Opus Coder, TDD)** on that branch:
  1. `depth_canon_make_cells.py` + regenerated `depth_canon_cells.json`: a new block `YAW-C4S` appended at the end
     (orders after 345): 24 rows named `exp25_YAW_C4_S40000_K{1,8}_s{43..46}_th{90,180,270}` (the names, metrics
     suffix `1_1.0_<cell>_rot<θ>` and file names the figure script expects), argv identical to the seed-42 template
     except `--seed` / `--rotate-deg` / `--eval-name`; plus ONE replay row `exp36_YAW_C4_S40000_K8_s42_th90` (argv
     identical to `exp25_YAW_C4_S40000_K8_s42_th90` except `--eval-name`). Every existing row byte-identical.
  2. `depth_canon_collect.py`: a separate `EXP36_BLOCKS = ("YAW-C4S",)` joins the no-flash decision (`EXP35_BLOCKS`
     itself unchanged, so no exp_35 default changes meaning; spawned through the committed overlay
     `overlays/no_flash/exp35_no_flash_launch.py`, receipt schema 2 with `attention_backend == sdpa`); a replay check
     `--exp36-replay-check` modelled on `exp35_ctl_check`: the new replay cell vs the archived exp_25 cell —
     `metrics` and `by_scene` strictly equal (`eval_FLAC._canon_strict_diff`), every other JSON key equal except
     `source_sha` / `eval_name`, the stream sidecar sha256 equal, the loaded checkpoint sha / bytes equal, the receipt
     backend `sdpa`; exit 0 equal / 1 mismatch / 2 refused. The reference digests (archived JSON, stream sidecar,
     receipt) are pinned in the code from the archived files.
  3. A copy step `--exp36-publish`: for each verified `YAW-C4S` seed-43–46 cell, copy its metrics JSON and stream
     sidecar into the exp-25 worktree's `outputs_FLAC/exp25_depth_canon/YAW/` (refuse if a name exists; never
     overwrite; never the replay cell); write a sha256 manifest of the copies into the exp_36 worklog.
  4. Tests (pytest, `src/tests/`), RED then GREEN, for the rows, the block membership, the replay comparator
     (equal / each mismatch kind / refused) and the publish refusal paths.
- **Dumps:** off for every A cell (`--no-dump-blocks YAW-C4S`, the replay included so its code path equals the 24's);
  the figure needs metrics only. Disclosed: the archived seed-42 cells dumped; dumping is a side output after scoring.
- **Run:** `EXPECT_HEAD=<A commit> bash depth_canon_eval_run.sh --only exp36_YAW_C4_S40000_K8_s42_th90 --no-dump-blocks
  YAW-C4S --lanes <one card>` → `--exp36-replay-check` → exit 0 required (HARD STOP otherwise: nothing else runs, report
  to Yixun) → `--blocks YAW-C4S` minus the replay (`--only` the 24 names) on the lane(s) the B schedule leaves free →
  per-cell verification by the runner → `--exp36-publish`. Each step's output teed to a timestamped log in
  `worklog/worklog_yixun/exp_36_yaw_c4_spread_claude/`.
- **GPU:** after PAUSED, A takes one card (the runner requires a card with no other compute process); if B's timing
  pass is not yet ready, A may use both lanes until then (the runner's per-cell lanes finish their current cell; no
  process is ever signalled; B's exclusive pass starts on a card only once that card is empty).
- **ETA (forecast):** replay ≈ 13 min; 24 cells ≈ 5.0 h on one exclusive card (12 × 11.9 + 12 × 12.6 min) or ≈ 2.5 h on
  two; plus the Coder round + review ≈ 1–1.5 h before the first launch.

- **Branch hygiene:** `exp-36-yaw-c4-spread` is a leaf — never merged into `exp-35-orbitrir-yaw`; its records and results reach `localization-exp` after exp_35's own merge (or as records-only commits); tests that pin the H35 manifest's row count / digest are updated on this branch only.

## 3. Hard stops
Replay mismatch or refusal; a receipt / backend failure (no sdpa evidence); a checkpoint sha mismatch; a publish
collision. Each stops A and is reported to Yixun; nothing is retried by hand.

## 4. Deliverables
The 24 metrics JSONs (+ stream sidecars) in `exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/YAW/`, the replay verdict
file, the copy manifest, results note `yaw_c4_spread_results.md` (per-cell table + the replay check), commits on
`exp-36-yaw-c4-spread` and the records on `localization-exp`. The figure script then needs `YAW_SEEDS = tuple(range(42, 47))`
(a one-line change in the paper repo — Yixun's or the desktop session's call; not done here).
