# exp_37 loc_efficiency — runbook B-a (GPU commands, NOT run by the Coder)

Written by the exp_37 Coder (Claude Opus 5.5, max effort), round B-a, for the Planner, who runs every command below
after the `claude_gpu_yield/` handshake. Plan: `plan_loc_efficiency.md` (§10 binding). Nothing here has been run.

## 0. Rules that bind every command

- **Checkout:** `~/codespace/exp-37-loc-efficiency`, branch `exp-37-loc-efficiency`, a clean tracked tree; record
  `git rev-parse HEAD` in every log (the block below does). Never commit in this checkout while a command runs.
- **One process at a time on GPU 0** (`CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`), each teed to a timestamped log in
  this folder; outputs under `$D/runs/<run>` (inside the worktree; every run needs a FRESH directory).
- **Every model command goes through the no-flash launcher** (`overlays/no_flash/exp37_no_flash_launch.py`) **and**
  passes `--require-attention-backend sdpa`: the launcher refuses (exit 86) unless the DiT resolves to SDPA, and the
  runner refuses unless this very process carries the launcher's hand-off; the run manifest then records
  `attention_backend: sdpa` and the overlay / launcher digests.
- **Fixed protocol flags:** `--candidate-batch-size 64 --tau 0.1 --score-sample-counts 1` (K_gen 1) and the bundle's
  timing flags `--synchronize-latency --warmup-query-count 1 --measure-core-forward` on every run (the headline
  timer is the bundle's uninstrumented region, pinned by `src/tests/test_exp37_headline_timer_guard.py`).
- **Mesh remap:** `--mesh-root AcousticRooms/room_mesh_obj_format` (every audited mesh is sha256-checked against the
  audit before use; the audit file is untouched; the digest map lands in the run identity).
- **B-F orbit plan (announcement 06):** the localization engine's FA path is per-angle — `cache_invariant_conditioning_branch`
  makes one conditioner call per frame angle (C4 = 0/90/180/270), eval mode; this branch has no
  `FRAME_AVG_MAX_FWD_SAMPLES` / shared-draw batching. P1 and YAW are evaluated `--cond-method vanilla`, B-F
  `--cond-method fa_invariant` (announcement 05: each arm under its own protocol).
- **Timing validity (plan §3, §10):** a timing is publishable only if its card held no other compute process for the
  whole run AND exp_36 has left the host (Yixun 15:30 order). Each run writes `host_records/start_*.json` and
  `end_*.json` (CPU affinity, torch intra/inter-op threads, OMP/MKL/BLAS variables, the card's UUID, `nvidia-smi`
  compute apps, top CPU processes); a run whose records show another compute app on its card is accuracy-only.
- **Hard stops (plan §9):** G0 FAIL or REFUSED; a launch refused for flash; a mesh sha mismatch (the runner refuses
  before writing anything); any timed pass with a co-tenant on its card (discard the timing, never publish it).

## 1. Shell set-up (once per shell)

```bash
cd /home/yixunhu/codespace/exp-37-loc-efficiency
export PY=/home/yixunhu/miniconda3/envs/flac/bin/python
export D=worklog/worklog_yixun/exp_37_loc_efficiency_claude
export E9=worklog/worklog_yixun/exp_09_localization_grid_preflight_claude
export LB=NeurIPS_2026_Workshop_Latency_Reproducibility
export L=overlays/no_flash/exp37_no_flash_launch.py
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
unset PYTHONPATH
set -o pipefail
# one model run: run_loc <run name> <ckpt> <cond method> <pilot manifest> <sample seed> [extra flags ...]
run_loc() {
  local run=$1 ckpt=$2 cond=$3 pilot=$4 seed=$5; shift 5
  local log=$D/loc_efficiency_$(date +%Y-%m-%d_%H-%M-%S)_${run}.log
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "REFUSED: the tracked tree is dirty"; return 2; }
  [ ! -e "$D/runs/$run" ] || { echo "REFUSED: $D/runs/$run exists (every run needs a fresh directory)"; return 2; }
  echo "HEAD $(git rev-parse HEAD) run=$run" | tee "$log"
  CUDA_VISIBLE_DEVICES=0 $PY $L localize_FLAC.py \
    --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json --ckpt-path "$ckpt" --cond-method "$cond" \
    --agree-ckpt weights/AGREE/AGREE_fullAR.pt --context-manifest $E9/context_manifest_exp01_seed42.json \
    --geometry-audit $E9/geometry_audit.json --pilot-manifest "$pilot" --dataset-root AcousticRooms \
    --mesh-root AcousticRooms/room_mesh_obj_format --output-dir $D/runs/$run --device cuda:0 \
    --candidate-batch-size 64 --sample-seed "$seed" --tau 0.1 --score-sample-counts 1 \
    --synchronize-latency --warmup-query-count 1 --measure-core-forward \
    --require-attention-backend sdpa "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}; echo "EXIT $rc" | tee -a "$log"; return $rc
}
```

`run_loc` is the only wrapper: it adds nothing but the HEAD / tree record, the tee and the exit code. Every command
below is `run_loc` with explicit arguments, so each is exact.

## 2. Pre-flight (CPU, ~1 min; before the first GPU command)

```bash
bash $D/exp37_assets.sh verify $PWD                               # expect: VERIFY PASS (4 digests, 16/16 meshes)
CUDA_VISIBLE_DEVICES= $PY $L --preflight                          # expect: one EXP37_NO_FLASH_LAUNCHER attention_backend=sdpa line, exit 0
CUDA_VISIBLE_DEVICES= $PY -m pytest -q -p no:cacheprovider src/tests/test_exp37_*.py   # expect: all passed
# CPU dry run of the G0 YAW invocation (CUDA hidden): the launcher, the hand-off, query 4467 and the 16 mesh
# digests are verified on the real inputs, then the runner stops at its CUDA requirement having written nothing.
CUDA_VISIBLE_DEVICES= $PY $L localize_FLAC.py --model-config src/configs/model_configs/FLAC/AR/FLAC_AR.json \
  --ckpt-path weights/exp20/YAW_40k.ckpt --cond-method vanilla --agree-ckpt weights/AGREE/AGREE_fullAR.pt \
  --context-manifest $E9/context_manifest_exp01_seed42.json --geometry-audit $E9/geometry_audit.json \
  --pilot-manifest $LB/data/selection/frozen_16room_128.json --dataset-root AcousticRooms \
  --mesh-root AcousticRooms/room_mesh_obj_format --output-dir $D/runs/cpu_preflight --device cuda:0 \
  --candidate-batch-size 64 --sample-seed 42 --tau 0.1 --score-sample-counts 1 --synchronize-latency \
  --warmup-query-count 1 --measure-core-forward --require-attention-backend sdpa --query-index 4467
#   expect: the sentinel line, then "RuntimeError: formal localization requires an available CUDA device" (exit 1)
#   and no $D/runs/cpu_preflight directory (done once by the Coder: loc_efficiency_2026-11-06_16-04-27_cpu_preflight_g0_YAW.log)
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv     # expect: nothing on GPU 0
```

## 3. Gate G0 (hard stop; ≈ 2–3 min per arm, ≈ 8 min total) — YAW first

Query 4467 (`single_channel_ir_1/Bathrooms/Bathrooms_idx_14/S009_R017_hybrid_IR.wav`, 27 candidates) of the bundle's
128-query selection (`--query-index 4467`; the warm-up query is 4467 itself), seed 42, compared with
`repeat_001/<arm>/queries/query_04467.npz`. Run each comparator only after its run exits 0; stop at the first
non-zero exit.

```bash
run_loc g0_YAW weights/exp20/YAW_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --query-index 4467
CUDA_VISIBLE_DEVICES= $PY tools/exp37_g0_compare.py --arm YAW --run-dir $D/runs/g0_YAW --output $D/g0/g0_YAW_verdict.json
#   exit 0 PASS -> continue; 1 FAIL -> STOP (environment / backend parity failed); 2 REFUSED -> STOP (fix the input)
run_loc g0_P1 weights/exp20/P1_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42 --query-index 4467
CUDA_VISIBLE_DEVICES= $PY tools/exp37_g0_compare.py --arm P1 --run-dir $D/runs/g0_P1 --output $D/g0/g0_P1_verdict.json --yaw-verdict $D/g0/g0_YAW_verdict.json
run_loc g0_BF weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42 --query-index 4467
CUDA_VISIBLE_DEVICES= $PY tools/exp37_g0_compare.py --arm BF --run-dir $D/runs/g0_BF --output $D/g0/g0_BF_verdict.json --yaw-verdict $D/g0/g0_YAW_verdict.json
#   P1 / BF: exit 1 with YAW PASS -> STOP that arm (wrapped-vs-extracted EMA parity suspect); report to Yixun
```

Per run ≈ 1–2 min start-up (FLAC + AGREE load, 16 mesh digests on the NAS, one room rebuild) + 2 × ~0.4 s.

## 4. B1 — K_ctx 8 headline latency, 128 queries (timing-valid only under §0's host gate)

The bundle's protocol: the frozen 16-room / 128-query selection, seed 42, batch 64, warm-up 1, one repeat. Run only
when exp_36 has left the host and GPU 0 is exclusive (else postpone, or run §5 first as accuracy). The AGREE selector
vector and the 3-arm summary are round B-b.

| # | command | expected wall |
|---|---|---|
| B1.1 | `run_loc b1_k8_P1_s42 weights/exp20/P1_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42` | ≈ 15 min (bundle Σ elapsed 839 s) + 1–3 min start-up |
| B1.2 | `run_loc b1_k8_YAW_s42 weights/exp20/YAW_40k.ckpt vanilla $LB/data/selection/frozen_16room_128.json 42` | ≈ 14 min (bundle 820 s) + 1–3 min |
| B1.3 | `run_loc b1_k8_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $LB/data/selection/frozen_16room_128.json 42` | ≈ 24 min (bundle 1,405 s) + 1–3 min |

Cross-check after B1.3 (Planner): OrbitRIR here vs the bundle's 10.919 s/query / 15.091 ms/candidate (host
difference reported, never hidden) — computed by the B-b summarizer once the selector vector exists.

## 5. B2 — 0.5 m candidate density, the 64-query pilot (P1 and B-F × seeds 42 / 43 / 44)

Accuracy runs; also timing runs when §0's host gate holds (B2's time axis is the query-micro mean over the 64
queries of `core_forward_total` + the shape-matched AGREE selector — never `elapsed_seconds`; plan §10).

| # | command | expected wall |
|---|---|---|
| B2.1 | `run_loc b2_0p50_P1_s42 weights/exp20/P1_40k.ckpt vanilla $E9/pilot_manifest_seed42_4_per_room.json 42` | ≈ 7 min (bundle 64-query Σ elapsed ≈ 420 s) + 1–3 min |
| B2.2 | `run_loc b2_0p50_BF_s42 weights/exp20/BF_40k.ckpt fa_invariant $E9/pilot_manifest_seed42_4_per_room.json 42` | ≈ 12 min (≈ 702 s) + 1–3 min |
| B2.3 | `run_loc b2_0p50_P1_s43 weights/exp20/P1_40k.ckpt vanilla $E9/pilot_manifest_seed42_4_per_room.json 43` | ≈ 7 min + 1–3 min |
| B2.4 | `run_loc b2_0p50_BF_s43 weights/exp20/BF_40k.ckpt fa_invariant $E9/pilot_manifest_seed42_4_per_room.json 43` | ≈ 12 min + 1–3 min |
| B2.5 | `run_loc b2_0p50_P1_s44 weights/exp20/P1_40k.ckpt vanilla $E9/pilot_manifest_seed42_4_per_room.json 44` | ≈ 7 min + 1–3 min |
| B2.6 | `run_loc b2_0p50_BF_s44 weights/exp20/BF_40k.ckpt fa_invariant $E9/pilot_manifest_seed42_4_per_room.json 44` | ≈ 12 min + 1–3 min |

Note for the Planner (not applied): the candidate noise seeds are keyed by (seed, query index, candidate index), so
B1.1 / B1.3 (seed 42, 128 queries) already contain the 64 pilot queries of B2.1 / B2.2 under identical seeds and
batches (only the warm-up query and run order differ); B2.1 / B2.2 could be read out of B1 if GPU time is short.

## 6. Order and totals

G0 (≈ 8 min) → B1.1–B1.3 when the host gate holds (≈ 60 min) → B2.1–B2.6 (≈ 70 min); if the host gate does not
hold yet after G0, run B2 first (accuracy) and B1 later. Total ≈ 2 h 20 min of GPU 0, one process at a time.
After each run: `EXIT 0` in the log, `run_manifest.json` with `attention_backend: sdpa`, `no_flash_launch` and
`mesh_root_override`, and `host_records/{start,end}_*.json` — read the co-tenant lists before using any timing.
