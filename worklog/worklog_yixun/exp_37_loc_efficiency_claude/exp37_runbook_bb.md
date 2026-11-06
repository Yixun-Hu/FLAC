# exp_37 loc_efficiency — runbook B-b (GPU commands, NOT run by the Coder)

Round B-b adds: the AGREE selector vector over any manifest, K_ctx = 1, the SEPARATE instrumented stage-split pass, and
the 3-arm summary. Everything in `exp37_runbook_ba.md` §0 (rules) and §1 (shell set-up, `run_loc`) applies unchanged:
one process at a time on GPU 0, every model run through the no-flash launcher with `--require-attention-backend sdpa`,
the bundle's flags, fresh output directories, timestamped tee'd logs, and publishable timings only under the host gate
(no other compute process on the card for the whole run, exp_36 gone from the host). Run after the B-a G0 PASS.

## 1. The AGREE selector vector (once; ≈ 5–6 min)

The selector term of every L_q. Arm- and K_ctx-independent (shape-only); the 128-query vector also serves every
64-query 0.5 m pilot (the summarizer accepts a superset vector and checks each row). Through the launcher for a uniform
log (AGREE has no flash path). Its own end-of-run host snapshot is judged like a run's.

```bash
LOG=$D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_selector_128.log
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "REFUSED: the tracked tree is dirty -- do not run"
{ echo "HEAD $(git rev-parse HEAD)"; CUDA_VISIBLE_DEVICES=0 $PY $L \
  tools/exp37_selector_latency.py --selection $LB/data/selection/frozen_16room_128.json \
  --context-manifest $E9/context_manifest_exp01_seed42.json --dataset-root AcousticRooms \
  --agree-checkpoint weights/AGREE/AGREE_fullAR.pt --device cuda:0 --candidate-batch-size 64 --repeat-count 3 \
  --output $D/selector/selector_128.json; echo "EXIT $?"; } 2>&1 | tee $LOG
# the tool refuses an existing --output (selector vectors are written once)
```

Expected: the bundle's selector summed 86.8 s of medians over the 128 queries → 3 repeats ≈ 4.5 min + AGREE load and
audio I/O ≈ 5–6 min. (CPU smoke on query 4467 done by the Coder: the CLI runs end to end on the real checkpoint.)

## 2. B1 K_ctx = 1 headline, 128 queries (host gate)

K_ctx = 1 feeds `contexts[0]` of the exp_09 manifest to the model; the candidate filter keeps the 8-context rule (the
27 / 92,608 frozen candidates are unchanged — tested on real data for query 4467); the run identity records the rule
and every query its context path.

| # | command | expected wall |
|---|---|---|
| B1.4 | `run_loc b1_k1_P1_s42 weights/exp20/P1_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --context-count 1` | ≈ 14 min + 1–3 min start-up (K8 minus ≈ 10 s of context encoding) |
| B1.5 | `run_loc b1_k1_YAW_s42 weights/exp20/YAW_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --context-count 1` | ≈ 14 min + 1–3 min |
| B1.6 | `run_loc b1_k1_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42 --context-count 1` | ≈ 23 min + 1–3 min (K8 minus ≈ 35 s) |

## 3. B1 stage split — the SEPARATE instrumented passes (host gate)

`--stage-split` runs `stage_split.execute_query_stage_split`: the headline's computation with CUDA-event stage marks
and per-segment allocator resets inside the timed windows (never the headline). Its core totals must agree with the
headline's within 2 % per query AND in aggregate, else the split is withheld (the headline never). K_ctx 8 first; the
K_ctx 1 splits complete the "3 arms × 2 K" table and may be dropped if GPU time is short (Planner's call).

| # | command | expected wall |
|---|---|---|
| B1.7 | `run_loc b1_split_k8_P1_s42 weights/exp20/P1_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --stage-split` | ≈ 15 min + 1–3 min (events add no sync; resets ≈ 0) |
| B1.8 | `run_loc b1_split_k8_YAW_s42 weights/exp20/YAW_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --stage-split` | ≈ 14 min + 1–3 min |
| B1.9 | `run_loc b1_split_k8_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42 --stage-split` | ≈ 24 min + 1–3 min |
| B1.10 | `run_loc b1_split_k1_P1_s42 weights/exp20/P1_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --context-count 1 --stage-split` | ≈ 14 min + 1–3 min |
| B1.11 | `run_loc b1_split_k1_YAW_s42 weights/exp20/YAW_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --context-count 1 --stage-split` | ≈ 14 min + 1–3 min |
| B1.12 | `run_loc b1_split_k1_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42 --context-count 1 --stage-split` | ≈ 23 min + 1–3 min |

## 4. The B1 summary (CPU, seconds)

```bash
CUDA_VISIBLE_DEVICES= $PY tools/exp37_summarize_latency.py --selection $LB/data/selection/frozen_16room_128.json \
  --selector $D/selector/selector_128.json \
  --run P1 8 $D/runs/b1_k8_P1_s42 --run YAW 8 $D/runs/b1_k8_YAW_s42 --run BF 8 $D/runs/b1_k8_BF_s42 \
  --run P1 1 $D/runs/b1_k1_P1_s42 --run YAW 1 $D/runs/b1_k1_YAW_s42 --run BF 1 $D/runs/b1_k1_BF_s42 \
  --split P1 8 $D/runs/b1_split_k8_P1_s42 --split YAW 8 $D/runs/b1_split_k8_YAW_s42 --split BF 8 $D/runs/b1_split_k8_BF_s42 \
  --split P1 1 $D/runs/b1_split_k1_P1_s42 --split YAW 1 $D/runs/b1_split_k1_YAW_s42 --split BF 1 $D/runs/b1_split_k1_BF_s42 \
  --output-json $D/b1/b1_latency_summary.json --output-md $D/b1/b1_latency_summary.md 2>&1 \
  | tee $D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_b1_summary.log
# exit 0: every row and the selector vector timing-valid; exit 3: some row is NOT PUBLISHABLE (co-tenant, unknown
# card, run never seen on its card, or a selector vector without an exclusive snapshot) -- the JSON / markdown say which.
```

Rows: s/query mean / median / P90 and ms/candidate dataset-wide / query median / P90 (the bundle's formulas; on the
bundle's own data they reproduce its expected summary exactly — `src/tests/test_exp37_latency_summary.py`); memory as
different quantities (whole-query allocated peak — bundle-compatible —, the selector's peak, their MAXIMUM as the
combined inference peak, the loaded-weight footprint after the FLAC and the AGREE loads); the stage split
(context / candidate conditioning / DiT / VAE decode / AGREE, dataset-wide ms per candidate) only when published.
Cross-check for the Planner: OrbitRIR K8 here vs the bundle's 10.919 s/query / 15.091 ms/candidate (another host).

## 5. Order and totals (GPU 0, one process at a time)

Selector (≈ 6 min) → B1.1–B1.3 K8 headline (runbook B-a, ≈ 60 min) → B1.4–B1.6 K1 headline (≈ 58 min) → B1.7–B1.9
K8 split (≈ 60 min) → B1.10–B1.12 K1 split (≈ 58 min, optional) → summary. ≈ 4 h with the K1 splits, ≈ 3 h without.
The plan §7 estimate (≈ 2 h for B1) predates the separate split pass that Codex BLOCKING 1 requires.
