# exp_35 orbitrir_yaw_evidence — research notes (read-only investigation)

By: Claude Opus 5.5 (max effort), research/planning subagent for the Planner, 2026-11-04 22:15–23:30 EST.
Read-only: no repo / NAS / env / process was modified; the only outputs are in `scratchpad/exp35/`.
Paths relative to `~/codespace/FLAC` unless stated. Checkouts read: FLAC `localization-exp` HEAD `7bff64b8`
(was `8f7bdbc0` at the start; the new commit is records-only), exp_25 worktree `~/codespace/exp-25-depth-canon`
(branch `exp-25-depth-canon`, HEAD `5b7e663b`), campaign pin `d6ca307a`. "collect.py" = `worklog/worklog_yixun/
exp_25_depth_canon_claude/depth_canon_collect.py`, "make_cells.py" / "runner" / "cells.json" likewise.

## 0. Findings that change the minimal plan (details in a–g)

1. **The exp_10 A6 rotated raws no longer exist.** Every 45/90/180/270 JSON of P1 and B-F named in
   `analysis/yaw_robustness/generated/yaw_{metrics,c4_metrics}_raw.csv` is absent (40+40 rotated rows: `exists=False`;
   only the 20+20 θ=0 rows resolve). Not in `outputs_FLAC/exp07_{P1,BF}/…/checkpoints/`, not in
   `A6000_METRICS_SHA256SUMS.txt`, not on the NAS endpoint folders, not in git, not under /home or /tmp. The paper repo
   already records it (`~/codespace/-Neurips-2026-workshop-Sound-Localization/fig_inputs/appendix/yaw_c4_off_orbit_provenance.json`
   note: "Original JSONs currently exist for 0 degrees only. Rotated values are recovered from archived per-seed CSVs").
   exp_25's ledger said the same (`depth_canon_comparator_ledger.md`: "no B-F@40k rotated cells on this box").
   Reuse = reuse of CSV values (4 decimals, the JSONs' own precision).
2. **exp_10 cells cannot feed the collector's estimators "exactly".** They have no `.stream.json` (no `input_hash`),
   no dumps, no `by_scene`, no receipt. `_arm_tier3` only uses input_hash-equal pairs (collect.py:1077-1078,
   1093-1099) → Δ_C4, the 45° probe and Δ_R are `None` for them; Tier 2 needs θ=0 dumps as references
   (collect.py:1019-1023). ⇒ B-F θ=0 T cells must be run at the new pin (10), not only R ×10 + T-dup.
   One-pin Δ_C4/45-probe for B-F additionally needs the C4+45 cells (8). Also: exp_10's B-F "C4 set" mixes
   evaluators — θ=0 = legacy loop (Aug 3) except K8 s42 = exp_11 batched backfill run on neuronic
   (`/n/fs/…` ckpt path, source `89f24cd5`); 45/90 = A6 (Aug 11-12), 180/270 = a6c4 (pin `069b72a5`, batched cap 64,
   per `analysis/yaw_robustness/validate_c4_cell.py:12,73-79`).
3. **The numerics backend changed after exp_25.** Yixun installed `flash-attn 2.7.4.post1` into the shared `flac` env on
   Nov 1 23:14 EST (`~/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/hostA/review/flash_attn_install_20261101.md`);
   `pip freeze | sha256sum` = `8b7a802e…` now vs `e896c48f…` in all 50 exp_25 launcher logs. The DiT attention
   (`src/models/transformer.py:12-18` import, `:429-437` flash path, casting fp32 q/k/v to fp16) is therefore NOT what
   exp_25 ran: every exp_25 cell log prints "No module named 'flash_attn' / flash_attn not installed". Eval logs by run
   date: no-flash ≤ Jul 17, **flash Jul 20 – Aug 18 (exp_07, exp_10; the A6/a6c4 cells of Aug 11–14 fall in this window, their logs are lost)**, no-flash Aug 19 – Oct 30 (exp_19/20/22/25),
   flash again since Nov 1. ⇒ the CODE eval path is unchanged since `d6ca307a` (below), the numerics are not.
   Note the exp_07↔exp_25 vanilla agreement (d) already spans a flash→no-flash switch.
4. **The kit hard-codes vanilla and one pin.** `make_cells.py:49-51` (COMMON_FLAGS `--cond-method vanilla`),
   `:111-114` (suffix with `cond_method="vanilla"`); `collect.py:123-125` (`record_reasons` expects
   `cond_method vanilla, frame_avg_angles None, orbit_execution "n/a", frame_avg_fwd_cap None`), `:455-458`
   (`cell_paths` → vanilla stem). A B-F cell would run but fail the runner's post-run verification (runner
   `:324-329`) → no receipt. One pin per collect (`record_reasons` `source_sha == pin` :125, manifest :328, receipt
   `expect_head` :412) and one campaign digest (`build_results` :1288-1290).
5. **The registered CANON composition control is not a valid Tier-2 reference for B-F**: its quarter-turn control
   p99 is exactly 0 (`depth_canon_cond_probe.json` control_p99.eval: d128/d256/d384 = 0; d64 0.0140, random 0.0110,
   pooled 0.0105) because canonical quarter turns are bit-exact, whereas a C4 rotation permutes FA's orbit sum
   (reassociated float additions + float32 pose composition) — exp_08 measured relL2 ≈ 0.0023 at 90/180/270 for the
   A-F FA checkpoint. Also `probe_reasons` binds the probe to the CANON@40k sha and `source_sha == pin`
   (collect.py:932-944) and the probe only knows canonicalization (`depth_canon_cond_probe.py:82-115, 225-290`).
6. **The main checkout cannot host a multi-hour exp_35 campaign**: the runner refuses/aborts on any HEAD move or dirty
   relevant tree (runner `:112-126`, `:421-426` "PIN DRIFT" exit 4), while the main session keeps committing records
   to `localization-exp` and exp_22's GPU kit (free-card watcher, command list pinned at `4576b562`) lives there.
   ⇒ run from a worktree (new branch from `localization-exp` HEAD, or the existing exp-25 worktree).
7. **A co-tenant lane roughly halves Yixun's step rate** (his own table job: 4.23 → 8.52 s/step; g).

## (a) How an exp_25 cell is defined and run

**Schema.** `cells.json` = header {schema_version, generator, plan, blocks, block_counts, common_flags, img_w 512,
seeds 42–46, dump_root `/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms`, farm_root `outputs_FLAC/exp25_depth_canon`,
splits {unseen 6337 / seen 6217}, arms {CANON, P1, YAW: model_config, model_config_sha256, trained_canon, ckpt}} +
230 rows with exactly 26 fields: cell, order, priority, block, subblock, arm, step, split, K, seed, rotate_mode,
rotate_deg, rotate_seed, canonicalize, allow_canon_mismatch, canon_in_play, canon_trained, canon_mismatch,
model_config(+_sha256), dataset_config(+_sha256), expected_stream_count, dump, metrics_suffix, eval_args.
Generated by `make_cells.py` (`_row` :89-123, `_pin_rows` :126-135, `build_manifest` :138-164); the runner refuses a
file that is not the generator's byte output (`make_cells.py --check` :182-186, runner :137).
Block counts: T 10, T-dup 2, C4+45 8, R 10, D-i 2, D-ii 2, S 10, P1-pin 28, YAW-pin 38, TRAJ 120.

**Exact evaluator argv** (example `exp25_P1_R_S40000_K8_s42`, from `depth_canon_2026-10-29_11-18-16_eval_plan.json`):
`CUDA_VISIBLE_DEVICES=<lane> HF_HUB_OFFLINE=1 /home/yixunhu/miniconda3/envs/flac/bin/python eval_FLAC.py
--model-config worklog/…/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json --dataset-config src/configs/dataset_configs/AR/eval/
acousticroom_unseeneval.json --cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64
--device cuda --record-per-scene --record-stream --expected-stream-count 6337 --seed 42 --rotate-mode random
--rotate-seed 42 --eval-name exp25_P1_R_S40000_K8_s42 --ckpt-path outputs_FLAC/exp25_depth_canon/P1/epoch=8-step=40000.ckpt
--dump-scored-waveforms /media/diskstation/yixunhu/FLAC/exp25_pred_waveforms` (runner `run_cell` :308-311; argv =
row eval_args + `--ckpt-path <farm hard link>` + dump root, collect.py `cell_argv` :387-390). Fixed cells carry
`--rotate-mode fixed --rotate-deg θ`; CANON cells add `--canonicalize`. num-workers = eval_FLAC default 4.

**Campaign launch** (worklog :1476/1608): from the worktree root, `EXPECT_HEAD=<40-hex pin> ALLOW_SHARED_GPU=1 setsid
nohup bash worklog/…/depth_canon_eval_run.sh --blocks <B,..> [--only C,..] [--limit N] [--lanes 0,1]
[--no-dump-blocks TRAJ]`. Gates: strict HEAD pin + committed relevant tree (eval_FLAC.py, src minus tests, data/AR,
the arm configs, the kit) re-hashed before every spawn and after every cell (:112-128, :421-426); config sha pins
(:129-135); NAS dump root (cifs) write probe (:154-176); checkpoint copied into a 0444 content-addressed cache
`<farm>/.ckpt_cache/<sha>.ckpt`, NAS MANIFEST line required, re-hashed before each spawn (:187-241, :433-436);
lane = ≥ 10,240 MiB free (:58) and, with `ALLOW_SHARED_GPU=1`, co-tenants disclosed not refused (:346-379);
resume = skip complete-and-verified cells, quarantine the rest (:259-296); receipt `<stem>.cell.json` only after
`--write-receipt --verify-after` (:324-329). Logs/plan JSON are teed into the exp_25 folder (LOGDIR = EXPDIR, :56).

**Yaw.** `prepare_eval_metadata` → `apply_rotation_plan` (eval_FLAC.py:1161-1217, 1231-1275): fixed = one integer
column roll `yaw_column_shift = round(α·512/2π) mod 512` (yaw_rotation.py:283-306; 45°→64, 90°→128, 135°→192,
225°→320 columns, all exact) of the depth panorama + float32 z-rotation of the poses (:309-326); context audio and
target RIR untouched; FA / canonicalization act after the rotation. Random = per-sample offsets d ~ U{0..511} drawn
batch-wise in item order from a dedicated CPU generator seeded with `--rotate-seed` = eval seed (eval_FLAC.py:2033-2038,
yaw_rotation.py:108-158), never touching the global RNG that `pl.seed_everything(seed)` (:2005) seeds for the per-batch
`torch.randn` noise (:2084). Verified: R and T cells of P1, CANON and YAW share `input_hash` (and R cells the
`assignment_hash`) for the same (K, seed) — e.g. K8 s42 input `4c950c1e…`, assignment `3df157e1…` — so B-F R/T cells
at the new pin must reproduce them (a free cross-campaign identity check).

**Per-cell wall times (exp_25, from cell-log birth→last write, which includes dump + receipt verification; tqdm loop in
brackets).** Phase 1 (dump cells, Oct 29–30, two lanes, both shared with a 32 GiB xRIR training): K1 median 17.6 min
(16.8 loop; n 22+20), K8 19.2 min (18.4; n 23+20); unshared lane (n 1–2): K1 12.8, K8 17.3–20.5. Phase 2 (TRAJ, no
dumps): GPU 0 shared K1 18.5 / K8 20.0; GPU 1 unshared K1 13.5 / K8 14.7 min. Footprint run 10:55–11:17 Oct 29:
1.9 GiB per lane (vanilla/CANON). Throughput two lanes ≈ 9.0 (phase 1) / 8.2 (phase 2) min per cell. Dump = 259.6 MB
per cell ([6337, 10240] float32); host RAM ≈ 743 MiB per cell at finalize (worklog :557). Collector `--collect` over 230
cells: 11 min CPU (collect log). Data are read from the NAS (`AcousticRooms` → cifs symlink), so cells are partly
IO-bound. No exp_25 B-F/fa cell exists. fa timing references (exp_10 logs, tqdm loop): Aug 3–4 next to the exp_10
FA training: fa K8 27.9–32.8 min, fa K1 17.4 min vs vanilla K8 12.5–14.1, K1 10.6–11.0 (ratio 2.2× K8, 1.6× K1);
Aug 5 unshared: fa K8 11.0–14.2, K1 7.3–8.4 (`fa_scratch_resume_2026-08-0{3,4,5}_*.log`). Yixun's own exp_30 table job
(fa a1 ≈ vanilla cost, next to his training): K8 15.6–15.8, K1 11.3–11.9 min.

## (b) Tiers as registered and as computed

Plan v15 §1 (`plan_depth_canon.md:28-31`), quoted:
- "**Tier 1 — structural (exact):** for every item of a rotated cell (C4, 45°, random yaw) the sha256 of the *canonical
  recovered-depth image* equals the one of the same item in the θ=0 cell with the same seed and K: 6,337 / 6,337 equal".
- "**Tier 2 — practical waveform invariance** … paired per-item relative L2 and L∞ between the scored waveforms of a
  rotated cell and its θ=0 pair (identical eval seed ⇒ identical generative noise and context draw, proven by identical
  `input_hash`) … `relL2 = ‖a−b‖₂ / max(‖b‖₂, 1e-6)` … Two controls: (i) the **determinism floor** — a duplicate θ=0
  cell (same seed, run twice; expected exactly reproducible) …; (ii) the **numerical-composition control**
  (`depth_canon_cond_probe.py`, GPU, fixed noise): for 64 unseen items × offsets {64, 128, 256, 384, random}, direct
  `canon(md)` vs `canon(rot_d(md))` compared stagewise … **Pre-registered PASS rule:** `p99(relL2) ≤ 3 × p99` of the
  composition control's waveform relL2 … **and** `p99(relL2) ≤ 1e-2` (hard cap); `max(relL2) ≤ 1e-1` is an alarm bound
  … the duplicate cell must be exactly reproducible (`max relL2 == 0`)".
- "**Tier 3 — metrics:** `Δ_C4(K)` = max−min over θ ∈ {0, 90, 180, 270}° (seed 42) and the 45° off-orbit probe;
  `Δ_R(K) = m_R − m_Z` seed-paired over seeds 42–46 … PASS if `Δ_C4` of CANON ≤ 10 % of P1's one-pin `Δ_C4` on every
  metric and `|Δ_R|` ≤ 1 eval-seed sd of `m_Z`."
- ε_cond = max fp32 conditioning relL2 of the control, registered ≤ 1e-5 (measured 8.12e-5 → FAIL; Yixun: report as
  registered + labelled diagnosis, adjudication A).

In collect.py: **T-dup** = block `T-dup` paired with its T cell (`make_pairs` :745-746), PASS iff input_hash equal,
max relL2 == 0 and dump sha256 equal (:1034-1041). **Composition control** = `load_control` (:962-986) of a registered
probe JSON validated by `probe_reasons` (:915-959; `PROBE_CONTRACT` :825-832), mapping 45°→d64, 90°→d128, 180°→d256,
270°→d384, random→random (`CONTROL_KEYS` :816); Tier 2 = every pair passes matched (≤ 3× its control p99) and cap
(≤ 1e-2) AND the pooled rule (:1005-1049; constants :815). **Δ_C4** = max−min of the item-weighted metric over the
four seed-42 θ cells, only if all four share the θ=0 input_hash (:1090, 1093, 1097); **45° probe** = v(45°) − v(0°),
seed 42, signed (:1098); **Δ_R** = mean over seeds 42–46 of v(R_s) − v(T_s), only when all 5 pairs are input_hash-equal
(:1094, 1099); **sd(m_Z)** = `statistics.stdev` (n−1) of the 5 θ=0 values (:1092, 1081-1084). **Item-weighted** = the
record's flat `metrics` block (`_value` agg "item" :1059-1063); per-scene = mean over the 10 `by_scene` families,
recorded only. Metrics: T60, C50, EDT, FD, R@1, R@5, R@10 (`TIER3_KEYS` :48-49; FD is in Tier 3 but not in Yixun's
six-metric tables). Rule: `pass_c4 = Δ_C4(arm) ≤ 0.10 × Δ_C4(P1-pin)`, `pass_r = |Δ_R(arm)| ≤ sd(m_Z arm)` for every
metric and both K, item-weighted (`tier3` :1103-1121, `TIER3_FRACTION` :1055); written only for CANON vs P1 today.
exp_25 outcome (`depth_canon_results.md`): Tier 1 PASS (20 pairs), Tier 2 FAIL as registered (C4 pairs exactly 0;
45° p99 0.0153/0.0152 and R p99 0.0143–0.0156 > cap; T-dup exact), Tier 3 PASS. P1-pin Δ_C4 (the yardstick):
K1 T60 0.9223, C50 0.1331, EDT 6.498, FD 0.0059, R@1 0.647, R@5 0.7101, R@10 1.073; K8 0.9033, 0.1337, 6.807, 0.0074,
0.5523, 0.7259, 1.089.

## (c) Evaluating B-F, and what Tier 2 means for it

- Checkpoint: NAS `ar_40k_endpoints/BF/epoch=8-step=40000.ckpt`, sha256 `5319feb4…` (MANIFEST.sha256 line present);
  local `weights/exp20/BF_40k.ckpt` hashes to the same; its embedded `model_config` == `exp_07_fa_scratch_claude/
  FLAC_AR_BF.json` (dict equality; file sha256 `3e677204…`), `training.cond_method fa_invariant`,
  `frame_avg_angles [0,90,180,270]`, no `frame_avg_max_fwd_samples` (trained on the legacy per-angle loop,
  announcement 06), no canonicalize / yaw_aug, global_step 40000 (verified 2026-11-04).
- Flags (announcements 05/06, explicit): `--cond-method fa_invariant --frame-avg-angles 0,90,180,270`
  (comma string, parsed at eval_FLAC.py:2282) `--frame-avg-max-fwd-samples 64` + the common flags. Record then holds
  `cond_method fa_invariant, frame_avg_angles [0.0,90.0,180.0,270.0], orbit_execution "batched", frame_avg_fwd_cap 64`
  (`orbit_provenance` :1304-1320, yaw_rotation.py:52,56); file stem gains `_fa_invariant_a4` (:630).
  Eval chunk plan: batch 64, cap 64 → 1 angle per forward for the 99 full batches; the 1-item tail batch (6337 = 99·64+1)
  runs its 3 rotated variants in one forward (yaw_rotation.py:563). `--canonicalize` + fa is refused (eval_FLAC.py:544-546).
- Runner support: the runner passes any row argv verbatim, so it needs **no change**. What blocks B-F is make_cells
  (vanilla flags + vanilla suffix) and the collector's vanilla expectations (finding 4). Minimal change = arm-aware
  protocol in both (a constant `{BF: fa_invariant, angles, "batched", 64}`; no new manifest field needed), new blocks,
  the `exp35_` name prefix, BF in `make_pairs` / `_index` / `tier2` / `tier3` / rendering, per-campaign pin. All of it is
  "the cell list + the collector change"; the runner, eval_FLAC.py and src/ stay untouched.
- **Tier 2 for B-F.** CANON's invariance is canonicalization (identical canonical input) and its control measures the
  float residue of `canon(rot_d(md))` vs `canon(md)`. B-F's invariance is the C4 frame average: exact in exact
  arithmetic for d ∈ {128, 256, 384} only, structurally absent at 45° (d = 64) and for ~99 % of random offsets
  (exp_10: B-F 45° probe T60 +1.45–1.57 vs sd 0.017–0.054). The faithful analog of the composition control is the same
  stagewise fixed-noise probe with A = fa(md), B = fa(rot_d(md)) on the B-F checkpoint, used as a control ONLY at
  d128/d256/d384 (at d64/random the probe would measure the structural effect, not a numerical floor). That needs a
  `--cond-method fa_invariant` switch in `depth_canon_cond_probe.py` (+ a BF `PROBE_CONTRACT`) — new code, ≈ 2–5 min GPU.
  Options for Yixun: (A, recommended) B-F probe, C4-only control; B-F 45°/R pairs tested against the pooled-C4 floor
  (deviation: the registered 45°→d64 / random→random mapping is meaningless for FA); (B, no new code) the registered
  CANON control as-is → B-F's C4 pairs must be bit-exact → expected FAIL caused by the reference; (C, no new code,
  descriptive) B-F's own full-split C4 pairs as the empirical floor for its 45°/R pairs. Under every option B-F's
  45°/R pairs fail Tier 2 by construction; T-dup is expected to be exact (eval forward passes are expected deterministic;
  exp_25's CANON T-dup K8 was bit-exact across GPU 0 → GPU 1). Tier 1 is undefined for B-F (no canonical image).

## (d) exp_10 A6 data and the θ=0 admissibility question

- **What exists.** `yaw_metrics_raw.csv` (60 rows: P1 "Vanilla P1@40k" and B-F "FA B-F@40k" × K{1,8} × {0,45,90} ×
  seeds 42–46) and `yaw_c4_metrics_raw.csv` (80 rows: × {0,90,180,270}); columns method, K, angle_deg, seed,
  source_json, T60, C50, EDT, FD, R@1/5/10 (GT), R@1/5/10 (geom); the shared 0/90 rows agree; summaries =
  mean + std(ddof=1), n = 5. Values = the JSONs' flat `metrics` block = item-weighted (`plot_yaw_curves.py:191,195`).
  Sources: θ=0 P1 = `exp07_P140_K{1,8}_s4*` + `exp07_P1_screen_S40000_ema` (K8 s42); θ=0 B-F = `exp10_BF40_*`
  (legacy loop, Aug 3) + `exp11_C4backfill_S40000_s42_K8` (batched, neuronic) for K8 s42 — `model_comparison.md`'s
  B-F row uses `exp07_BF40_fae_rot0` (legacy loop, A6000) for K8 s42 instead (`gen_model_comparison.py:84-85`);
  the two K8 s42 records differ by T60 0.0012, EDT 0.0011, R@1/R@5 ±0.047. Rotated: A6 45/90 (Aug 11–12, pin
  unrecorded), a6c4 180/270 (Aug 14, pin `069b72a5`, batched cap 64, run from `/tmp/flac-a6c4-069b72a`). All
  num_samples 6337, batch 64, EMA, cfg 1, 1 step, bf16 cond autocast, flash-attn era. Raws lost (finding 1).
- **Vanilla θ=0, exp_07/exp_10 (flash era) vs exp_25 P1-pin (`d6ca307a`, no flash), 5-seed means:**

  | K | T60 | C50 | EDT | FD | R@1 | R@5 | R@10 |
  |---|---|---|---|---|---|---|---|
  | 1 exp_10 | 10.2870 | 1.0884 | 43.4371 | 0.3186 | 4.9897 | 15.1113 | 22.5091 |
  | 1 exp_25 | 10.2871 | 1.0884 | 43.4369 | 0.3186 | 4.9708 | 15.1049 | 22.5248 |
  | 1 max per-seed abs Δ | 0.0003 | 0.0001 | 0.0031 | 0.0001 | 0.0947 | 0.1263 | 0.0474 |
  | 8 exp_10 | 8.9927 | 1.0093 | 40.6496 | 0.3218 | 5.1728 | 15.4300 | 23.4086 |
  | 8 exp_25 | 8.9926 | 1.0093 | 40.6495 | 0.3218 | 5.2075 | 15.4711 | 23.4496 |
  | 8 max per-seed abs Δ | 0.0004 | 0.0000 | 0.0033 | 0.0001 | 0.0947 | 0.0789 | 0.1736 |

  **Explicitly: T60, C50, EDT (and FD) agree to three decimals in the 5-seed mean; R@1/R@5/R@10 do NOT** (mean Δ
  −0.019/−0.006/+0.016 pp at K1, +0.035/+0.041/+0.041 pp at K8; per seed up to 0.17 pp = 11 of 6,337 items; one item
  = 0.0158 pp). Same picture for the rotated seed-42 cells (exp_10 A6/a6c4 vs exp_25 P1-pin C4+45, 8 cells): T60 ≤ 0.0002,
  C50 ≤ 0.0001, EDT ≤ 0.0018, FD ≤ 0.0001; R@k up to 0.158 pp. Δ_C4 / 45° probe of P1 from both sources agree on the
  acoustic metrics (e.g. K1 ΔT60 0.9223 vs 0.9223, probe45 +0.6827 vs +0.6827), differ on retrieval (R@1 0.584 vs 0.647).
- **Would the exp_25 evaluator score B-F at θ=0 differently from the exp_10 legacy loop?** Code: same orbit arithmetic,
  same angle order, same divisor; eval mode has no RoPE draw, so loop and batched "agree numerically"
  (yaw_rotation.py:43-44, 541-542; exp_11 probe "eval-bf16 deviation 0", exp_10 worklog :90); same seeding
  (`pl.seed_everything` + per-batch `torch.randn`; FA consumes no global RNG in eval) and same item-weighted `metrics`.
  Differences: (i) the 1-item tail batch is regrouped (eval_FLAC.py:1500-1503, 1529-1530); (ii) the attention
  backend (exp_10 flash; exp_25 no flash; exp_35 flash unless shadowed — finding 3); (iii) evaluator commit (exp_10
  records carry no `source_sha`). Expected effect: inside the vanilla envelope above (acoustic ~1e-4–3e-3, retrieval
  ± a few items). ⇒ No systematic difference is predicted, but the 2 measurement cells cost nothing extra: they are
  the first two of the 10 B-F T cells that Δ_R and Tier 2 need anyway. Reference values (seed 42): K1
  `exp10_BF40_K1_s42` T60 9.4859, C50 1.0547, EDT 41.2371, FD 0.3284, R@1 5.2391, R@5 15.7014, R@10 23.7494;
  K8 `exp07_BF40_fae_rot0` 8.1902, 0.9804, 38.8113, 0.3333, 5.3022, 16.3958, 24.2386 (and `exp11_C4backfill`
  8.1890, 0.9805, 38.8102, 0.3332, 5.3495, 16.3484, 24.2386).
- exp_10 B-F spreads from the CSV (seed 42, mixed provenance, NOT the registered estimator): Δ_C4 K1 T60 0.0016,
  C50 0.0002, EDT 0.0037, FD 0.0000, R@1 0.0158, R@5 0.0316, R@10 0.0947; K8 0.0013, 0.0001, 0.0044, 0.0001, 0.0631,
  0.0474, 0.0158 — vs 10 % of P1-pin: everything passes except **K8 R@1 (0.0631 > 0.0552, i.e. 4 items vs 3.5)**; K1
  R@10 is close (0.0947 vs 0.1073). 45° probe K1 T60 +1.567, C50 +0.108, EDT +2.247, R@1 −1.373, R@5 −3.409,
  R@10 −4.466; K8 +1.453, +0.112, +2.093, −1.610, −4.071, −4.434. sd(m_Z) K1 0.0543 / 0.0040 / 0.3475 / R 0.166,
  0.241, 0.150; K8 0.0172 / 0.0016 / 0.0740 / R 0.064, 0.057, 0.164. ⇒ registered Tier 3 for B-F: C4 rule on the
  retrieval metrics is within one or two item flips of the threshold; R rule is expected to FAIL by construction.

## (e) Reusable exp_25 cells (pin `d6ca307a`, no flash; artifacts in the exp-25 worktree, dumps on the NAS)

| arm | θ=0 (T) | 45° | 90/180/270 | random (R) |
|---|---|---|---|---|
| CANON | K1/K8 × s42–46 (+ T-dup s42) | s42 only | s42 only (+ TRAJ) | K1/K8 × s42–46 |
| P1-pin | K1/K8 × s42–46 | s42 only | s42 only | K1/K8 × s42–46 |
| YAW-pin | K1/K8 × s42–46 | s42 only | s42 only | K1/K8 × s42–46 |

Only the 30 θ=0 metric JSONs + receipts + sidecars are force-added in git (FLAC checkout and exp-25 branch); the R / C4
artifacts live untracked in `~/codespace/exp-25-depth-canon/outputs_FLAC/exp25_depth_canon/` (CANON 622, P1 93, YAW
85 files) and the dumps in `exp25_pred_waveforms` (90 dirs). Values (5-seed mean ± sd, item-weighted): random yaw —
P1 K1 T60 11.033±0.035, C50 1.1714, EDT 46.613, R@1 4.248, R@5 13.779, R@10 20.944; K8 9.742, 1.0960, 43.911, 4.510,
14.262, 21.663; CANON K1 9.788, 1.1049, 44.783, 5.062, 15.522, 23.188; K8 8.419, 1.0273, 42.450, 5.211, 15.903, 23.904.
45° seed 42 — CANON K1 9.805 / 1.1049 / 44.289 / 5.097 / 15.623 / 23.276, K8 8.429 / 1.0255 / 42.454 / 5.160 / 15.954 /
23.813. **Gap for the figure and table (b): CANON 45° exists for seed 42 only** (P1 and B-F have 5 seeds at 45°
from exp_10) → 8 cells (CANON × 45° × K × s43–46) for 5-seed error bars, or a single-seed CANON 45° point.

## (f) plot_yaw_curves.py (`analysis/yaw_robustness/`, identical copy in cylindrical-dinov3)

Constants `LEGACY_ANGLES (0,45,90)`, `C4_ANGLES (0,90,180,270)`, `SEEDS`, `METHODS ("Vanilla P1@40k","FA B-F@40k")`
(:16-19); six-panel `C4_FIGURE_METRICS` (:56-63); `--mode legacy|c4|all` (:79-84); `paths_for` hard-codes the
exp07/exp10/exp11/a6/a6c4 filenames per method/K/angle (:95-160); `collect` loads JSONs, checks seed and rotate_deg,
emits raw rows + summary (mean, std ddof=1) (:163-207); `lookup` requires the summary's angles == requested angles
(:217-235); `plot_grid` = 2 methods × 2 K, colors #2E63E7 / #F2762E, markers o / s, K=1 dashed hollow, K=8 solid,
legend "Vanilla, K=k" / "FA B-F, K=k", suptitle + "(5 seeds, @40k training steps)", png+pdf 300 dpi (:238-324);
`write_readme` has a quoting bug (generated README.md contains literal `"` fragments) (:327-345). Because the rotated
raws are gone, `--mode legacy|c4|all` can no longer regenerate on this box (FileNotFoundError at :89-90). Paper labels
(`…/scripts/plot_yaw_off_orbit.py:31`): "Vanilla P1@40k" → FLAC, "FA B-F@40k" → OrbitRIR. **Minimal extension:**
`OFFORBIT_ANGLES = (0, 45, 135, 225)`; a source table per (model, K, angle, seed): θ=0 and 135/225 from metric JSONs
(exp_25 T / exp_35 cells), FLAC/OrbitRIR 45° from the committed `yaw_metrics_raw.csv` rows (validated against its
summary CSV, as the paper script's `refresh_data` does), CANON 45° from exp_25 C4 th45 s42 + exp_35 s43–46; a third
style (CANON, e.g. green triangle); labels FLAC / OrbitRIR / CANON; `--mode offorbit` writing
`yaw_offorbit_metrics_{raw,summary}.csv` + `yaw_offorbit_metrics.{png,pdf}`; per-JSON checks (rotate_deg, seed,
cond_method, n_samples 6337, source_sha ∈ {d6ca307a, exp_35 pin}). Existing modes untouched.

## (g) GPU state and Yixun's training (read-only)

22:22 EST Nov 4: both A6000 100 % util, each holds one rank of Yixun's exp_30 s43 training (pids 3027719 / 3028196,
cwd `~/codespace/flac-fork-attn`, 8,248 MiB each); free 40,270 / 40,080 MiB. Queue (`exp30_attn_queue.sh:239`):
trainings s44 (done Nov 4 06:11) then s43 (started 06:12:55); after s43 its eval worker runs the s43 table job on
GPU 1 (40 cells); nothing else queued. s43 at 22:33 EST: global step 9,685/40,000. Step rate from the tqdm elapsed
field of `train_exp30_ACEattn_s43_20261104T061255_741947.log`: **unshared 4.218–4.238 s/step per 500-step window
(epoch 1, 19,380 s / 4,550 steps incl. validation)**; **during his own s44 table job on GPU 1 (06:12:59–16:01:40,
40 cells, `queue_20261101T233607_741947.log`): 7.61–9.71 s/step, mean 8.52 s/step over steps 0–4,000 (×2.01)**.
(HANDOFF: the eqtail co-tenant training slowed s44 from ≈ 3 h to ≈ 6.8 h per 2,500 steps, ×2.27.) Unshared ETA of s43:
30,315 steps × 4.23 s ≈ 35.6 h → ≈ Nov 6 10:10 EST (+ ≈ 15 min validation). His table job cost him ≈ 4.9 h of training
progress (9.8 h × (1 − 4.23/8.52)); every hour of our co-tenant lane costs ≈ 0.5 h.
