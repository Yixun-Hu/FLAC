# exp_37 loc_efficiency — runbook B-c (GPU commands, NOT run by the Coder)

> **SUPERSEDED for execution (fix round 1, 2026-11-06):** run `exp37_runbook.sh` per `exp37_runbook_r2.md`. This file is kept as the history of its round. Codex code review 1 found commands here that it no longer accepts: no waveform dumps or replays, snapshot-only co-tenancy, unenumerated seeds, masked exit statuses.

Round B-c adds the B2 candidate-density points (0.75 m runs, the dedicated nested-1.0 m and coarse-to-fine timing
passes), B3 (inference orbit {0} / {0,180} / C4 × θ {0, 90}; orbit latency), and the aggregation + Pareto figure.
`exp37_runbook_ba.md` §0 (rules) and §1 (shell set-up) apply unchanged; run after G0 PASS (runbook B-a §3). All
expected walls are scaled from the bundle's measured per-candidate costs on the same GPU / CPU model (P1 ≈ 9.1 ms and
B-F ≈ 15.2 ms per candidate incl. scoring) plus 1–3 min start-up; they are estimates.

## 1. One more wrapper (same body as `run_loc`, with an audit and a timing switch)

```bash
# run_loc_x <run> <ckpt> <cond> <pilot> <seed> [extra ...]
#   AUDIT=<geometry audit> (default the exp_09 audit); TIMING=0 drops the bundle's timing flags (rotated runs only)
run_loc_x() {
  local run=$1 ckpt=$2 cond=$3 pilot=$4 seed=$5; shift 5
  local audit=${AUDIT:-$E9/geometry_audit.json} timing="--synchronize-latency --warmup-query-count 1 --measure-core-forward"
  [ "${TIMING:-1}" = 0 ] && timing=""
  local log=$D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_${run}.log
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "REFUSED: the tracked tree is dirty"; return 2; }
  [ ! -e "$D/runs/$run" ] || { echo "REFUSED: $D/runs/$run exists (every run needs a fresh directory)"; return 2; }
  echo "HEAD $(git rev-parse HEAD) run=$run audit=$audit timing=${TIMING:-1}" | tee "$log"
  CUDA_VISIBLE_DEVICES=0 $PY $L localize_FLAC.py \
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --ckpt-path "$ckpt" --cond-method "$cond" \
    --agree-ckpt weights/AGREE/AGREE_fullAR.pt --context-manifest $E9/context_manifest_exp01_seed42.json \
    --geometry-audit "$audit" --pilot-manifest "$pilot" --dataset-root AcousticRooms \
    --mesh-root AcousticRooms/room_mesh_obj_format --output-dir $D/runs/$run --device cuda:0 \
    --candidate-batch-size 64 --sample-seed "$seed" --tau 0.1 --score-sample-counts 1 $timing \
    --require-attention-backend sdpa "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}; echo "EXIT $rc" | tee -a "$log"; return $rc
}
P050=$E9/pilot_manifest_seed42_4_per_room.json
A075=$D/audit_0p75/geometry_audit_0p75.json; P075=$D/audit_0p75/pilot_manifest_0p75_seed42_4_per_room.json
N100=$D/manifests/nested_1m_pilot_seed42_4_per_room.json
```

Inputs built on CPU and committed: the 0.75 m audit (`5c58e39c…`, global branch pinned to z_band — computed
full_height —, 5,337 / 5,337 grids non-empty) and its pilot (`07a71e3c…`, the same 64 queries in the same order,
11,537 pairs); the nested-1.0 m manifest (`c80eb12e…`, 6,610 pairs). The parameterized audit tool reproduces the
frozen 0.5 m audit with 0 differences beyond host paths (`loc_efficiency_2026-11-06_16-49-51_audit_0p50_reproduction.log`).

## 2. Selector vectors for the 0.75 m and nested counts (host gate; ≈ 4 min)

```bash
for sel in 0p75:$P075 nested_1m:$N100; do name=${sel%%:*}; selection=${sel#*:}
  { echo "HEAD $(git rev-parse HEAD)"; CUDA_VISIBLE_DEVICES=0 $PY $L tools/exp37_selector_latency.py \
    --selection $selection --context-manifest $E9/context_manifest_exp01_seed42.json --dataset-root AcousticRooms \
    --agree-checkpoint weights/AGREE/AGREE_fullAR.pt --device cuda:0 --candidate-batch-size 64 --repeat-count 3 \
    --output $D/selector/selector_$name.json; echo "EXIT $?"; } 2>&1 \
    | tee $D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_selector_$name.log
done      # ≈ 2 min (0.75 m) + ≈ 1.5 min (nested); the 0.5 m points use selector_128.json (runbook B-b §1)
```

## 3. B2 — 0.75 m runs (accuracy; timing-valid under the host gate)

| # | command | expected wall |
|---|---|---|
| B2.7 | `AUDIT=$A075 run_loc_x b2_0p75_P1_s42 weights/exp20/P1_40k.ckpt vanilla $P075 42` | ≈ 2 min + start-up |
| B2.8 | `AUDIT=$A075 run_loc_x b2_0p75_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $P075 42` | ≈ 3.5 min + start-up |
| B2.9 | `AUDIT=$A075 run_loc_x b2_0p75_P1_s43 weights/exp20/P1_40k.ckpt vanilla $P075 43` | ≈ 2 min + start-up |
| B2.10 | `AUDIT=$A075 run_loc_x b2_0p75_BF_s43 weights/exp20/BF_40k.ckpt fa_invariant $P075 43` | ≈ 3.5 min + start-up |
| B2.11 | `AUDIT=$A075 run_loc_x b2_0p75_P1_s44 weights/exp20/P1_40k.ckpt vanilla $P075 44` | ≈ 2 min + start-up |
| B2.12 | `AUDIT=$A075 run_loc_x b2_0p75_BF_s44 weights/exp20/BF_40k.ckpt fa_invariant $P075 44` | ≈ 3.5 min + start-up |

Shared coordinates get different generation noise at 0.75 m (seeds keyed by the grid index; disclosed, seeds 42–44
mitigate it). The 1.0 m and c2f ACCURACY comes offline from the 0.5 m runs (B2.1–B2.6) — no GPU.

## 4. B2 — dedicated nested-1.0 m and c2f timing passes (seed 42; host gate)

| # | command | expected wall |
|---|---|---|
| B2.13 | `run_loc_x b2t_nested_P1_s42 weights/exp20/P1_40k.ckpt vanilla $P050 42 --candidate-subset nested_1m` | ≈ 1 min + start-up |
| B2.14 | `run_loc_x b2t_nested_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --candidate-subset nested_1m` | ≈ 2 min + start-up |
| B2.15 | `run_loc_x b2t_c2f_P1_s42 weights/exp20/P1_40k.ckpt vanilla $P050 42 --candidate-subset c2f --c2f-top-k 5` | ≈ 1.7 min + start-up |
| B2.16 | `run_loc_x b2t_c2f_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --candidate-subset c2f --c2f-top-k 5` | ≈ 3 min + start-up |

The nested pass is the headline executor on the integer subset with the full 0.5 m indices for the seeds; the c2f pass
times the actual sequential path (observed once → context → coarse gen + AGREE → top-5 + Chebyshev-0.5 m neighbours →
only the new refined candidates gen + AGREE → union argmax). Neither is ever resumed (fresh directory only).

## 5. B3 — OrbitRIR inference orbit (localization accuracy; 64 queries, seeds 42–44)

C4 θ0 = the B2 0.5 m B-F runs (B2.2 / B2.4 / B2.6). θ0 runs keep the timing flags (harmless; not used); θ90 runs are
accuracy-only (`TIMING=0`; the rotation executor refuses timing).

| # | command (×3: `s42` / `s43` / `s44` with seed 42 / 43 / 44) | expected wall each |
|---|---|---|
| B3.1–3 | `run_loc_x b3_o0_t0_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --frame-avg-angles 0` | ≈ 7 min + start-up |
| B3.4–6 | `run_loc_x b3_o0_180_t0_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --frame-avg-angles 0 180` | ≈ 8.5 min + start-up |
| B3.7–9 | `TIMING=0 run_loc_x b3_o0_t90_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --frame-avg-angles 0 --rotate-deg 90` | ≈ 8 min + start-up |
| B3.10–12 | `TIMING=0 run_loc_x b3_o0_180_t90_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --frame-avg-angles 0 180 --rotate-deg 90` | ≈ 10 min + start-up |
| B3.13–15 | `TIMING=0 run_loc_x b3_c4_t90_s42 weights/exp20/BF_40k.ckpt fa_invariant $P050 42 --rotate-deg 90` | ≈ 13 min + start-up |

(θ90 adds a CPU rotation of every candidate item, ≈ 1–2 ms each.)

## 6. B3 — orbit latency, B1-style (128 queries, K_ctx 8; host gate)

| # | command | expected wall |
|---|---|---|
| B3.16 | `run_loc_x b3_lat_o0_s42 weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42 --frame-avg-angles 0` | ≈ 14 min + start-up |
| B3.17 | `run_loc_x b3_lat_o0_180_s42 weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42 --frame-avg-angles 0 180` | ≈ 17 min + start-up |

C4's latency is B1.3 (runbook B-a).

## 7. Aggregation + the Pareto figure (CPU, seconds)

Write `$D/b2b3/b2b3_spec.json` (paths relative to the worktree) and run the aggregator:

```json
{"pilot_0p50": "<E9>/pilot_manifest_seed42_4_per_room.json",
 "pilot_0p75": "<D>/audit_0p75/pilot_manifest_0p75_seed42_4_per_room.json",
 "nested_manifest": "<D>/manifests/nested_1m_pilot_seed42_4_per_room.json",
 "selector_0p50": "<D>/selector/selector_128.json", "selector_0p75": "<D>/selector/selector_0p75.json",
 "selector_nested": "<D>/selector/selector_nested_1m.json",
 "b2_runs": [{"arm": "P1", "grid": "0.50", "seed": 42, "dir": "<D>/runs/b2_0p50_P1_s42"}, "... P1 / BF x 0.50 / 0.75 x 42 / 43 / 44 (12 entries)"],
 "b2_timing": [{"arm": "P1", "kind": "nested_1m", "dir": "<D>/runs/b2t_nested_P1_s42"}, {"arm": "BF", "kind": "nested_1m", "dir": "<D>/runs/b2t_nested_BF_s42"},
               {"arm": "P1", "kind": "c2f", "dir": "<D>/runs/b2t_c2f_P1_s42"}, {"arm": "BF", "kind": "c2f", "dir": "<D>/runs/b2t_c2f_BF_s42"}],
 "b3_runs": [{"orbit": "0", "theta": 0, "seed": 42, "dir": "<D>/runs/b3_o0_t0_s42"}, "... orbits 0 / 0_180 / c4 x theta 0 / 90 x seeds (18 entries; c4 theta 0 = <D>/runs/b2_0p50_BF_s4x)"],
 "b3_latency": {"selection": "<LB>/data/selection/frozen_16room_128.json", "selector": "<D>/selector/selector_128.json",
                "runs": [{"orbit": "0", "dir": "<D>/runs/b3_lat_o0_s42"}, {"orbit": "0_180", "dir": "<D>/runs/b3_lat_o0_180_s42"},
                         {"orbit": "c4", "dir": "<D>/runs/b1_k8_BF_s42"}]},
 "external": {"accuracy": "worklog/worklog_yixun/exp_23_five_method_64_random_fallback/summary.json",
              "latency_csv": "<LB>/expected/latency_per_query.csv"}}
```

```bash
CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=$D/b2b3/.mpl $PY tools/exp37_aggregate_b2b3.py --spec $D/b2b3/b2b3_spec.json \
  --output-json $D/b2b3/b2b3_summary.json --output-md $D/b2b3/b2b3_summary.md --figure-stem $D/b2b3/b2_pareto 2>&1 \
  | tee $D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_b2b3_aggregate.log
# exit 3 = some timing is not publishable (co-tenant, unknown card, missing pass) -- the tables say which
```

The aggregator refuses any run whose identity is not the claimed arm / grid / seed / orbit / θ / pilot (and binds the
nested pass to its manifest). The c2f k ∈ {1, 3, 10} rows are post-primary sensitivity (accuracy only).

## 8. Not in this runbook

- **B3 synthesis cells** ({0} / {0,180} × θ {0, 90} × K {1, 8}, seed 42, eval_FLAC): Codex's split puts them on a
  separate H35-compatible leaf of exp_36's evaluator (closure proved unchanged) — not this branch.
- **B4** (P1@87.5k): no step-87,500 checkpoint was found in the searched local / NAS roots; skipped per Yixun's
  conditional instruction.

## 9. Order and totals (GPU 0, one process at a time)

Selectors (≈ 4 min) → B2.7–12 (≈ 30 min) → B2.13–16 (≈ 16 min) → B3.1–6 (≈ 60 min) → B3.7–15 (≈ 1 h 50 min) →
B3.16–17 (≈ 35 min) → aggregation. ≈ 4 h 15 min for B-c; with B-a (≈ 2 h 20 min) and B-b (≈ 3–4 h) ≈ 9.5–10.5 h of
GPU 0. The accuracy-only θ90 runs (≈ 1 h 50 min) could move to GPU 1 once exp_36 is done, but only while GPU 0 runs
no timing pass (CPU co-tenancy would perturb it) — the Planner's call.
