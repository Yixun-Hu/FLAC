> **STATUS: PLAN r4 — APPROVED by Codex plan r4 (12:01 EST 2026-11-05; `orbitrir_yaw_evidence_codex_plan_r4_review.md`; its NIT applied to §13's disclosure). §14 = implementation amendments A1–A2 (Planner, 2026-11-05 14:36:36 EST; within the approved plan; Codex checks them in code review C3). Written as PLAN r4 (2026-11-05 11:59:16 EST) = r3 + §13 (Codex plan r3 = REVISE with ONE blocker — the GPU-1 FIFO exclusion — resolved by an explicit, disclosed risk acceptance by the Planner, the third option Codex named; three NITs). r3 header: PLAN r3 (2026-11-05 11:53:19 EST) = r2 + §12 (answering Codex plan r2 = REVISE: atomic GPU ownership, the supervisor state machine, the probe chunk, child imports; still no new decision for Yixun). r2 header: PLAN r2 (2026-11-05 11:45:40 EST) = r1 + §11 (the binding implementation contract answering Codex plan r1 = REVISE; no new decision for Yixun, per Codex). r1 header: PLAN r1 (2026-11-05 11:30:08 EST; Planner edits 2026-11-05 11:34:50 EST: §2b decisions, single-cell chunks) — Yixun's decisions of 2026-11-05 applied; Codex plan r0 findings addressed; awaiting Codex plan re-review r1.** Nothing runs on a GPU before Codex accepts r1, the code rounds pass review and the integrative review passes. Written by an Opus 5.5 planning subagent for the Planner. Evidence: `orbitrir_yaw_evidence_research_notes.md`; r0 review: `orbitrir_yaw_evidence_codex_plan_r0_review.md`.

# plan_orbitrir_yaw_evidence — exp_35 (r1)

## 1. Goal and frozen scope
OrbitRIR (B-F) gets yaw-invariance evidence to the exp_25 standard, so CANON vs OrbitRIR can be stated on two axes: quality at θ=0 and invariance. No training.
- **Checkpoint:** NAS `ar_40k_endpoints/BF/epoch=8-step=40000.ckpt`, sha256 `5319feb4…2328`, 723,922,667 B. Its embedded config equals `exp_07_fa_scratch_claude/FLAC_AR_BF.json` (sha `3e677204…a328`).
- **Evaluator:** the exp_25 recipe — EMA, cfg 1.0, 1 Euler step, batch 64, bf16 conditioning autocast, `--expected-stream-count 6337`, `--record-per-scene --record-stream`, scored dumps on every cell. Attention runs through torch SDPA via the committed no-flash overlay (§3b).
- **B-F flags:** `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64`.
- **Chunk plan (announcement 06):** one angle per forward on the 99 full batches; the 1-item tail batch runs its 3 rotated variants in one forward. Eval mode takes no RoPE draw.

| phase | block (draft rows: scratch `cells_draft.json`, exact exp_25 26-field schema, orders 230–329) | cells |
|---|---|---|
| (i) GPU 1 co-tenant | **CTL**: P1, θ=0, s42, K1/K8 — evaluator identity, E1 hard stop | 2 |
| (i) | **BF-pin T**: θ=0, K1/K8 × s42–46. Gives m_Z, sd(m_Z), the Δ_R partners and the Tier 2 references; the s42 cells are Yixun's 2 measurement cells (E2) | 10 |
| (i) | **BF-pin R**: random yaw, K1/K8 × s42–46 | 10 |
| (i) | **BF-pin C4**: 90/180/270/45, K1/K8, s42 | 8 |
| (i) | **BF-dup**: θ=0 s42 re-run, K1/K8 | 2 |
| (i) | **B-F probe**: the Tier 2-C4 reference (§5) | 1 run |
| (ii) free cards only, after s43 ends and after the Planner's slowdown report | **OFF-P1** / **OFF-BF** / **OFF-CANON** (135/225 × K × s42–46) and **CANON-45** (45°, s43–46, mandatory) | 20 / 20 / 20 / 8 |
| (ii), only if E2 FAILS | P1 and B-F at 45°, K1/K8 × s43–46 (s42 already exists at one pin) | 16 |

Deliverables go to `orbitrir_yaw_results.{md,json}`:
- **(a) θ=0 table:** FLAC (exp_25 P1-pin T), CANON (exp_25 T) and OrbitRIR (BF-pin T at H35). Columns T60, C50, EDT, R@1/5/10; K1 and K8; mean ± sd over seeds 42–46, item-weighted. The historical B-F row (legacy loop, flash era) is printed beneath as a reference only.
- **(b) The same six metrics at 45° and under random yaw.**
  - 45°: FLAC and OrbitRIR from the A6 CSV unless E2 fails; CANON from exp_25 s42 plus CANON-45.
  - Random yaw: FLAC and CANON from exp_25 R; OrbitRIR from BF-pin R.
- **(c) Tier 3 spread table** with a B-F row next to CANON / P1-pin / YAW-pin (those three taken from the exp_25 JSON). Also B-F Tier 2-C4, T-dup, and the 45° and random-yaw dump distances (descriptive).
- **Figure:** `analysis/yaw_robustness/generated/yaw_offorbit_metrics.{png,pdf}` plus `_raw.csv` and `_summary.csv` (§6).

## 2. Yixun's decisions of 2026-11-05 → where applied
1. **Exact-standard phase (i)** (30 B-F + 2 CTL + probe, no CSV splice) → §1, §4 (no exp_10 value enters Tier 2 or Tier 3).
2. **Committed, checksummed no-flash stub** inside the runner digest; receipts record the backend; exact CTL equality is a hard stop → §3b, §3c.
3. **Tier 2-C4 is OrbitRIR's claim**; 45° and random-yaw distances are descriptive → §4.
4. **The A6 CSV `0ae554c0…` controls E2** → §4.
5. **CANON-45 ×8 is mandatory**; FLAC and OrbitRIR 45° come from the CSV with per-point provenance unless E2 fails → §1, §6.
6. **GPU 1 co-tenancy for phase (i) only**, with the unrounded 9.712 / 8.521 guard and one cell per chunk at first. Figure cells go on free cards after s43 ends, after a measured-slowdown report → §7.
7. **exp_22 uses the same no-flash backend**; its watcher never pre-empts the exp_35 lane → §7, §10.

## 2b. Planner decisions (2026-11-05, within Yixun's "proceed without asking")
1. The overlay lives in a new top-level `overlays/no_flash/` (outside `src/`, so no training-closure gate sees it), shared with exp_22's backend round.
2. The card reservation is a file `/home/yixunhu/codespace/.gpu_reservations/<GPU UUID>.json` (outside every checkout); the exp_22 watcher must honour it.
3. The E2-fail contingency is 16 cells: s42 at 45° already exists at one SDPA pin per model (P1: exp_25 C4+45 at `d6ca307a`; B-F: this phase's C4+45 at H35); Codex's 20 counted without that reuse.
4. Single-cell chunks throughout phase (i) (not 2-cell chunks after the first four): the guard can then never let a second cell start after a violation (≈ +30 min).
5. The exp_25 R and 45° cells are external evidence: verified offline at their own pin and force-added with the records at the end (§5).

## 3. r0 → r1
| Codex r0 finding | r1 resolution | implementation site |
|---|---|---|
| P1 BLOCKING minimal scope | Frozen at 28 Tier-3 cells + 2 dup + 2 CTL + the probe. The 22-cell CSV splice is dropped (decision 1). | `depth_canon_make_cells.py:39`, `:138-164` |
| P2 HIGH arm-aware protocol | One per-arm protocol table (vanilla default; BF = fa_invariant, angles `[0.0, 90.0, 180.0, 270.0]`, `batched`, cap 64, bf16). It drives argv, the `_fa_invariant_a4` suffix, record validation, probe binding and rendering. Guards: the exp_25-only render must equal the committed `depth_canon_cells.json` byte for byte (sha `1073ef64…0575`); the exp_25 mode must rewrite `depth_canon_results.json` byte for byte (sha `ec3fcd08…b6e0`). | make_cells `:49-51`, `:89-123`; collect `:123-129`, `:453-461`, `:1264-1266`; tests §9 |
| P3 BLOCKING Tier 2 semantics | Tier 2-C4 is registered (§4) against B-F's own probe at d128 / d256 / d384. 45° and R are descriptive. Tier 3 unchanged. | collect `:1005-1049`, `:816-832`, `:915-986`; probe `:46-47`, `:82-115`, `:174-212`, `:225-296`, `:347-421` |
| P4 BLOCKING backend provenance | Committed, sha-pinned overlay inside the runner digest, with a pre-flight and a per-cell sentinel check; receipt v2 records the backend; exact CTL equality is a hard stop with no fallback (§3b, §3c). | runner `:116-117`, `:129-138`, `:308-331`; collect `:372-379`, `:393-440`, `:624-670`, `:1402-1477` |
| P5 HIGH E2 source | One reference: the A6 CSV `yaw_metrics_raw.csv` (sha `0ae554c0…bbfb23da`), rows FA B-F@40k / angle 0 / seed 42, per K. Units explicit (§4). | collect: new `--exp35-e2` |
| P6 HIGH campaign separation | `--exp35` mode applies the §5 fail-closed list. | collect: new exp35 functions; `main :1547-1574` |
| P7 HIGH figure | CANON-45 ×8 is mandatory; every point carries provenance columns (§6). | `plot_yaw_curves.py:16-19`, `:66-85`, new mode |
| P8 BLOCKING GPU plan | One GPU 1 co-tenant lane. Single-cell chunks THROUGHOUT (Planner decision 4, §2b). Guard thresholds from his log: 500-step ≤ 9.712, running mean ≤ 8.521, 100-step early warning ≤ 9.950. Card reservation; phase (i) only; ETA (§7). | `exp35_training_guard.py` (round 6) |
| P9 BLOCKING pre-registration | E1–E5 rewritten with expectation and verdict separated; FD stays in the Tier 3 verdict (§4); frozen launch checklist (§8). | §4, §8 |
| P10 BLOCKING exp_22 | Reopened in an **exp_22 round, not exp_35 code**: its tests, gates, probe and passes put the same committed overlay on PYTHONPATH, and record the backend from the dispatch, not from `find_spec` (`eval_localization.py:719,863` misreports under a shadow). Its free-card watcher stays suspended until then and honours §7's reservation. | §7, §10 |

### 3b. The no-flash overlay (decision 2)
**The file.** `overlays/no_flash/flash_attn/__init__.py`, in a new top-level directory with no `__init__.py`, so `[tool.setuptools.packages.find]` never ships it. Exact content, sha256 `0d0e52ca693e236a2e0a44cd61f61539fdf995cff5132d1f621181a0f3e7c023`, verified in scratch:
```
"""No-flash overlay (exp_35; Yixun 2026-11-05 decision 2). Shadows the installed flash_attn so that
src/models/transformer.py:12-18 falls back to torch SDPA, the attention backend exp_25 ran with.
Put on PYTHONPATH by depth_canon_eval_run.sh only; never installed, never imported elsewhere."""
raise ImportError("exp35 no-flash overlay: flash_attn disabled, attention backend = torch SDPA")
```

**What it does (verified in scratch on CPU).** Under it, `find_spec("flash_attn")` resolves to the overlay and the import raises the sentinel. Importing `src.models.transformer` prints the sentinel, then `flash_attn not installed, disabling Flash Attention` (`transformer.py:15-16`), and leaves `flash_attn_func is None`, so the SDPA path runs (`:439-440`) instead of the flash branch (`:429-438`). `src` still resolves to the checkout.

**Runner changes:**
- `overlays/no_flash` joins RELEVANT (`:116-117`), so the campaign, spawn and post digests and the dirty check cover it.
- A constant `NO_FLASH_SHA256` is checked at start.
- A CPU pre-flight runs before any spawn and must confirm three things: the `find_spec` origin is the overlay file, the import raises exactly the sentinel, and `src` comes from `<repo>/src`.
- Every evaluator spawn (`:311`) runs with `PYTHONPATH=<repo>/overlays/no_flash`; any inherited PYTHONPATH is dropped.

**Per cell (no eval-path change).** After rc 0 the cell log must contain exactly one sentinel line, immediately followed by `flash_attn not installed, disabling Flash Attention`, and no `No module named 'flash_attn'`. Otherwise the cell FAILS, gets no receipt, and the campaign hard-stops. Receipt schema 2, for exp_35 blocks only, appends `attention_backend: "sdpa"` and `no_flash_overlay_sha256`. exp_25's schema-1 receipts are verified exactly as before. `verify_cell` re-reads the receipt's log for the sentinel pair.

### 3c. E1 — CTL hard stop, before any B-F cell
For each K, `exp35_P1_T_S40000_K{k}_s42` at H35 must equal `exp25_P1_T_S40000_K{k}_s42` at d6ca307a. The reference is the force-added receipt `outputs_FLAC/exp25_depth_canon/P1/epoch=8-step=40000_metrics_1_1.0_exp25_P1_T_S40000_K{k}_s42.cell.json` (sha K1 `fa1ea729…`, K8 `54490e2a…`). Equal means all of:
1. The dump waveform sha256 (manifest `sha256` and the COMPLETE marker): K1 `8cb47c945d55…e8f8`, K8 `1aace782f222…0fd3`.
2. Dump `n` 6337, `float32`, `[6337, 10240]`, and the same `input_hash`.
3. The stream-sidecar sha256: K1 `31aa02dcd5dc…793d`, K8 `8d8018cabfd8…4af6`. This implies input_hash K1 `37d2f5693fa6…` / K8 `4c950c1e2ed0…` and assignment `4894d0e119f7…`.
4. The metrics record: `metrics` and `by_scene` type-strictly equal, and every other key equal except `source_sha` and `eval_name` (`ckpt_path` is the same farm link).
5. `ckpt_sha256_loaded` `c4c67882…` and `ckpt_bytes_loaded` 723,927,531.
6. Receipt v2 `attention_backend == "sdpa"`.

The check is `depth_canon_collect.py --exp35-ctl-check K` (exit 0 equal / 1 mismatch / 2 refused). Any mismatch stops the campaign with a report; no B-F cell runs.

## 4. Pre-registration (frozen before any cell)
- **E1:** §3c. Exact, no fallback.

- **E2.** It decides only the 45° source, never phase (i). Per K at seed 42, the new `exp35_BF_T_S40000_K{k}_s42` is compared with the A6 CSV row (`yaw_metrics_raw.csv`, sha `0ae554c0…`, method `FA B-F@40k`, angle 0, seed 42):

  | K | T60 | C50 | EDT | FD | R@1 | R@5 | R@10 | source |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 9.4859 | 1.0547 | 41.2371 | 0.3284 | 5.2391 | 15.7014 | 23.7494 | `exp10_BF40_K1_s42` (legacy loop, A6000, flash era) |
  | 8 | 8.1890 | 0.9805 | 38.8102 | 0.3332 | 5.3495 | 16.3484 | 24.2386 | `exp11_C4backfill_S40000_s42_K8` (batched, neuronic, flash) |

  - PASS iff all 14 hold: |ΔT60| ≤ 0.001 (T60 error, %), |ΔC50| ≤ 0.0005 dB, |ΔEDT| ≤ 0.005 ms, |ΔFD| ≤ 0.0005, |ΔR@k| ≤ 0.2 percentage points.
  - PASS → FLAC and OrbitRIR 45° come from the CSV.
  - FAIL → the +16 phase-(ii) cells run, and 45° comes from one-pin SDPA cells (s42: exp_25 P1-pin C4 th45 and exp_35 BF C4 th45).
  - Disclosed: the K8 reference is a neuronic batched cell, 0.0012 in T60 off the A6000 legacy record `exp07_BF40_fae_rot0`. Seeds 43–46 are compared descriptively.

- **Tier 2-C4** (OrbitRIR's invariance claim; a verdict).
  - Pairs: (K1, K8) × (90°, 180°, 270°) at s42, each against BF-pin T s42 with equal input_hash.
  - Reference: the B-F probe's eval-protocol waveform relL2 p99 at the matched offset (90°→d128, 180°→d256, 270°→d384), and pooled over the three offsets.
  - PASS iff every pair's p99(relL2) ≤ 3 × the matched control p99, AND the pooled-C4 p99 (all items of the six C4 pairs) ≤ 3 × the pooled control p99 (all probe items at the three offsets), AND every pair's p99 ≤ 1e-2.
  - Alarm: any C4 item with relL2 > 0.1 suspends the claim until item triage.
  - T-dup (K1, K8): max relL2 must be 0 and the dump sha256 equal (the determinism requirement).
  - A control p99 of exactly 0 demands exact equality, as registered for exp_25.
  - 45° and random-yaw pairs: relL2 / relL∞ summaries only — outside the claim, no verdict. B-F ε_cond (fp32, C4 offsets) is reported without a verdict.

- **Tier 3** (verdict; registered rule, unchanged; item-weighted).
  - Definitions: Δ_C4 = max−min over 0/90/180/270 at s42 (all four input_hash-equal). 45° probe = v45 − v0 at s42. Δ_R = mean over s42–46 of v(R) − v(T) (all five pairs input_hash-equal). sd(m_Z) = the n−1 SD of the five θ=0 values.
  - `pass_c4`: Δ_C4(BF) ≤ 0.1 × Δ_C4(P1-pin). `pass_r`: |Δ_R(BF)| ≤ sd(m_Z BF). Both are checked for T60, C50, EDT, **FD**, R@1, R@5, R@10 at K1 and K8 — 28 checks, FD in the verdict.
  - P1-pin Δ_C4 comes from the exp_25 JSON: K1 0.9223 / 0.1331 / 6.498 / 0.0059 / 0.647 / 0.7101 / 1.073; K8 0.9033 / 0.1337 / 6.807 / 0.0074 / 0.5523 / 0.7259 / 1.089.
  - No exp_10 value enters.

- **E3 (the registered `pass_c4` rule).** The verdict is the rule's. Non-binding expectation: it passes on T60, C50, EDT and FD; the retrieval rows are uncertain.

- **E4 (expectation, kept separate from the verdict).** Expected: Tier 2-C4 PASS, T-dup exact; Tier 3 overall FAIL through `pass_r`, because B-F is C4-invariant only (the A6 45° probe was T60 +1.45 at K8 and +1.57 at K1); large 45° and R distances.

- **E5 (descriptive only, no verdict; after phase (ii)).**
  - B-F at 135° and 225° vs B-F at 45° (135 = 45+90, 225 = 45+180): same-pin s42 pairs, metric differences and dump relL2.
  - CANON at 135° and 225° vs CANON at 0°.
  - P1 vs angle.

## 5. Collector and probe
**`--exp35` collect mode** (`depth_canon_collect.py`) selects only exp_35 blocks at `--pin H35`. It refuses unless all of the following hold:
1. **The external exp_25 JSON:** hash `ec3fcd08…b6e0`, `schema_version` 1, `pin` d6ca307a…, `partial` true with `missing` exactly the 20 registered S-block names, `invalid` == {}.
2. **Every reused field is finite and non-null:**
   - tier3 for CANON / P1 / YAW × K × item × 7 metrics: `n_z` 5, `n_r_pairs` 5, and `mean_z`, `sd_z`, `delta_c4`, `probe45`, `delta_r`;
   - the θ=0 `table` stats with n 5.
3. **Reused rows stay under `external_reference.exp25`** and never pair with H35 cells.
4. **All H35 cells verify** with receipt v2, under exactly one campaign digest.
5. **Every stream matches the registry** `exp35_exp25_stream_registry.json` (frozen by a reviewed helper from exp_25's P1, CANON and YAW artifacts, which agree across arms).
   - Every B-F and OFF stream reproduces the registered input_hash per K/seed.
   - R cells reproduce the assignment_hash per seed; fixed cells reproduce the angle's assignment_hash.

   | stream | prefixes (12-hex here; full values in the registry) |
   |---|---|
   | T input, K1, s42–46 | `37d2f5693fa6 df150ea93d68 5709f434b246 fb4df53656ca 7d139a5679ed` |
   | T input, K8, s42–46 | `4c950c1e2ed0 1c048da9c6af 8398cd8655ad ef2c937139f8 163f89a4b84a` |
   | R assignment, s42–46 | `3df157e1df19 ea95d3cb08c8 fc3baefa5903 9d69ba364e6e bdc1eeba2fe0` |
   | fixed 0/90/180/270/45 | `4894d0e119f7 1d76e861d6eb a2db1d990eae 2edc90b917cb 943070e7d264` |
6. **The sha-pinned A6 CSV is read only by E2 and the figure.**
7. **CTL equality holds** (§3c).

The exp_25 R and 45° cells needed for (b) and the figure are external cells. They are verified offline at their own pin with `verify_cell(pin=d6ca307a, dump_check="receipt", check_link=False)` against the exp-25 worktree's artifacts, and force-added with the records at the end.

The default (exp_25) mode is unchanged: with `--blocks` set to the 10 exp_25 blocks it must rewrite `depth_canon_results.json` byte for byte. This is a validation-ladder step on CPU, ≈ 15 min, with NAS reads:
`CUDA_VISIBLE_DEVICES= python …/depth_canon_collect.py --collect --pin d6ca307a85b9b0a75ec9d47f4d237e63a3db527e --no-dump-blocks TRAJ --partial --blocks T,T-dup,C4+45,R,D-i,D-ii,S,P1-pin,YAW-pin,TRAJ --base-dir ~/codespace/exp-25-depth-canon --out-json <scratch>.json --out-md <scratch>.md`
The output JSON must have sha256 `ec3fcd08…b6e0`.

**B-F probe** (`depth_canon_cond_probe.py --arm BF`):
- **Pairs:** A = FA(md) with no canonicalization; B = FA(rotate_d(md)) for d ∈ {128, 256, 384}. FA = `invariant_conditioning(conditioner, md, device, (0.0, 90.0, 180.0, 270.0), max_fwd_samples=64)` under each protocol's conditioning autocast.
- **Stages and protocols:** the same four stages (cond, velocity, latent, waveform) and the same fixed per-item noise. Protocols fp32 (autocast off, matmul highest) and eval (bf16, medium), plus the determinism floor (A run twice).
- **Items:** the first 64 of the unseen K=8 loader (`acousticroom_unseeneval.json`, sha `063c66c2…`); seed 42, batch 64, cuda.
- **Binding:** checkpoint `5319feb4…` with loaded bytes 723,922,667; config `FLAC_AR_BF.json` `3e677204…` == embedded; the overlay sentinel present in its log.
- **Registration and output:** registered only at H35 on a clean tree. Output `orbitrir_bf_cond_probe.json` + `.sha256`, untracked until the phase ends. Verdict = masks equal and determinism exact.
- The collector's `PROBE_CONTRACT_BF` mirrors all of this.

## 6. Figure (`plot_yaw_curves.py --mode offorbit`, after phase (ii))
**Layout.** Angles 0 / 45 / 135 / 225; six panels; FLAC / OrbitRIR / CANON, with CANON in a third style; K=1 dashed hollow, K=8 solid; mean ± sd (ddof 1) over s42–46.

**Sources per angle:**
- θ=0: FLAC from exp_25 P1-pin T, OrbitRIR from BF-pin T (H35), CANON from exp_25 T.
- 45°: FLAC and OrbitRIR from the A6 CSV if E2 passes, otherwise from one-pin cells; CANON from exp_25 s42 plus CANON-45.
- 135 / 225: the phase-(ii) cells.

**`_raw.csv` columns:** method, K, angle_deg, seed, campaign (exp10_A6 | exp25 | exp35), evaluator_pin (40-hex | `unrecorded`), attention_backend (flash | sdpa), orbit_execution (loop | batched | n/a; A6 45° is batched, inferred from the run date after `1479304`), source_path, source_sha256 (the JSON, or the CSV sha plus row key), then the metrics. The summary CSV holds n, mean and std.

The existing modes are untouched. They cannot regenerate here, because the rotated raws are lost.

## 7. GPU run, guard, ETA
**Where.** Worktree `~/codespace/exp-35-orbitrir-yaw`, new branch `exp-35-orbitrir-yaw`, at H35 (the reviewed code commit).
- Assets as in the exp-25 worktree: `AcousticRooms` → NAS; `weights/{AGREE,FLAC/VAE.ckpt,…}` as symlinks to the main checkout. The runner stages the farm cache (P1, BF, CANON@40k, ≈ 2.2 GB).
- No commit lands in the worktree during a phase; records go to `localization-exp`.

**Launch command.** `EXPECT_HEAD=H35 ALLOW_SHARED_GPU=1 bash worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh --lanes 1 --only <chunk>`

**Chunk sequence (33 single-cell chunks; Planner decision 4), with the check after each:**
1. CTL K1 → E1
2. CTL K8 → E1
3. BF T s42 K1 → E2
4. BF T s42 K8 → E2
5. The probe
6–33. Twenty-eight single-cell chunks: T s43–46, R, dup, C4+45. They start only if the guard held through chunks 1–4. A violation stops after the running cell (the runner is never signalled), so stop latency is at most one cell.

**Guard.** Read-only, on Yixun's `train_exp30_ACEattn_s43_20261104T061255_741947.log`, through a reviewed parser `exp35_training_guard.py`. s/step comes from the tqdm elapsed field within an epoch; windows align to the epoch's step counter; epoch-boundary validation is excluded. STOP if any of these holds:
- any completed 500-step window > 9.712;
- the mean since our first spawn > 8.521;
- any completed 100-step window > 9.950 (early warning: his own table job's worst 100-step window over s43 steps 0–4,000).

For reference: unshared windows are 4.210–4.280 (100-step) and 4.218–4.244 (500-step); the baseline is 4.23. The guard is polled every ≤ 5 min and between chunks.

**Hard stops** (no further spawn; report to Yixun): E1 mismatch; a guard violation; any receipt or verification failure (including the overlay sentinel); a failed overlay pre-flight; an invalid probe.

**Reservation.** Before the first chunk the Planner writes `/home/yixunhu/codespace/.gpu_reservations/GPU-6ac2604a-e84e-0615-1221-ccbed7c268da.json` (holder exp_35, branch, H35, since) and removes it after the phase. The exp_22 free-card watcher (suspended; built later) must treat a reserved card as busy and never take GPU 1 while the lane runs.

**When his s43 training ends.** His queue then starts its s43 table job on GPU 1. The remaining phase-(i) chunks move to `--lanes 0` once GPU 0 has no compute process; the lane never co-tenants with his table job. Phase (ii) runs without `ALLOW_SHARED_GPU`, so only on free cards.

**ETA for phase (i), counted from the start of the first cell, co-tenant on GPU 1: ≈ 16.5 h (14–21 h).**
- CTL 17.6 + 19.2 min.
- B-F: 15 × K1 at ≈ 24 min (20–30) plus 15 × K8 at ≈ 35 min (30–45).
- Probe ≈ 10 min.
- Gates and checks over 33 chunks ≈ 70 min (single-cell chunks; ≈ +30 min vs 2-cell chunks).

It is re-estimated after chunks 3–4, and is shorter if his training ends first. Each co-tenant hour costs him ≈ 0.5 h of training (≈ 8 h if phase (i) overlaps his training throughout; his own table job cost ≈ 4.9 h). His s43 was at step 20,499 / 40,000 at 11:20 EST Nov 5, running ≈ 4.235 s/step; unshared it ends ≈ Nov 6 10:15 EST.

**Phase (ii)** (68 cells): ≈ 19 h on one free card, ≈ 9.5 h on two. If E2 fails, +16 cells ≈ 2.2 h on two.

## 8. Frozen pre-launch checklist (in the launch entry, before the first cell)
- **Code state:** H35, branch, worktree path, clean `git status`. `git diff --stat d6ca307a H35 -- eval_FLAC.py src ':(exclude)src/tests' data/AR` should touch only `src/localization/*`.
- **Cells:** cells.json sha and 330 rows; `make_cells.py --check`; the exp_25-only render sha `1073ef64…`; the sha of the 100 exp_35 rows; the phase-(i) cell order.
- **Checkpoints and configs:** P1 `c4c67882…`, BF `5319feb4…`, CANON@40k `52c64b52…` (CANON manifest `1e481447…`, NAS MANIFEST lines); configs BVp1 `733ca52b…`, BF `3e677204…`, CANON `aba94032…`; datasets `48753740…` / `063c66c2…`.
- **Assets and environment:** VAE `6198ff0d…`, AGREE_fullAR `3a13243d…`, the DINOv3 HF snapshot; overlay `0d0e52ca…` and its pre-flight output; flac python, torch 2.7.0+cu126, flash-attn 2.7.4.post1 (present, shadowed), pip-freeze `8b7a802e…`.
- **External references:** exp_25 JSON `ec3fcd08…`; the CTL receipts and hashes of §3c; the stream registry sha; the A6 CSVs `0ae554c0…` / `e4602115…` / `441ade8b…` / `b21ffbe7…`; the CANON probe `ac2f8baf…` (reference only).
- **Contracts and gates:** the B-F probe contract; the stop gates (E1; guard 9.712 / 8.521 / 9.950; receipt failures).
- **GPU:** lane GPU 1 by UUID, free memory ≥ 10,240 MiB, the co-tenant pids, the reservation file.
- **His training:** the log path, its epoch/step, and the current 500-step rate.

## 9. Per-file implementation
TDD in `src/tests/`; commits under 200 code lines; Codex code review per round, then an integrative review.

**Round 1 — backend.**
- Files: new `overlays/no_flash/flash_attn/__init__.py`. Runner: RELEVANT += `overlays/no_flash` (`:116-117`); `NO_FLASH_SHA256` and the pre-flight after `:138`; PYTHONPATH at `:311`; at `:324-329`, the sentinel check and `--attention-backend sdpa --no-flash-overlay-sha256 <sha>`.
- Tests (`test_depth_canon_eval_run.py` harness): sentinel present → receipt v2 with sdpa; sentinel absent or `No module named` → FAILED, no receipt; edited overlay → refused or PIN DRIFT; inherited PYTHONPATH dropped.

**Round 2 — generator.**
- Files: `depth_canon_make_cells.py` gets the BF arm (existing arm schema) and per-arm flags and `cond_method` (`:49-51`, `:89-123`); `exp35_` names; blocks CTL, BF-pin, BF-dup, OFF-P1, OFF-BF, OFF-CANON, CANON-45 (priorities 10–16, orders 230–329; `:39`, `:138-164`); regenerate `depth_canon_cells.json`.
- Tests: the exp_25-only render equals the `1073ef64…` bytes; 330 rows; counts and order; the BF argv and suffix; names and suffixes injective, including against exp_25; `--check`.

**Round 3 — collector protocol and receipt v2.**
- Files: `ARM_PROTOCOL`; `record_reasons` (`:123-129`) and `cell_paths` (`:453-461`); `RECEIPT_KEYS` v2 (`:372-379`), `receipt_reasons` (`:393-440`), `build_receipt` (`:624-670`), CLI (`:1420-1431`, `:1448-1477`); the log re-check.
- Tests: a valid BF record passes and each wrong field is refused; v1 receipts are unchanged; `test_gen_model_comparison_depth_canon.py` stays green.

**Round 4 — probe.**
- Files: `--arm BF` per §5 (`:46-47`, `:82-115`, `:174-212`, `:225-296`, `:347-446`, CLI).
- Tests (toy stack): an invariant toy gives C4 relL2 0; no d64 or random for BF; BF refuses canonicalization; any deviation sets registered=false.

**Round 5 — exp35 collect, two commits.**
- 5a: external reference, the registry and its helper, `--exp35-ctl-check`, `--exp35-e2`.
- 5b: Tier 2-C4, the Tier 3 BF row, tables (a)/(b)/(c), and rendering into the exp_35 folder. Touches `make_pairs :732-759`, `_index :1066-1074`, `tier2 :1005-1049`, `tier3 :1103-1121`, `table_block :1124-1135`, `render_md :1321-1394`, `main :1547-1574`.
- Tests: every §5 fail-closed condition; a poisoned CSV leaves Tier 2/3 unchanged; a registry mismatch is refused.

**Round 6 — figure and guard.**
- Files: `plot_yaw_curves.py --mode offorbit` and `exp35_training_guard.py`.
- Tests: `test_plot_yaw_offorbit.py` and `test_exp35_training_guard.py`, with fixture logs that include epoch boundaries.

**Validation ladder:**
1. CPU suites: depth_canon_*, gen_model_comparison, and the new tests.
2. The exp_25 byte-for-byte collect.
3. Runner `DRY_RUN=1` at H35: overlay pre-flight, plan of the 32 phase-(i) cells.
4. Codex integrative review.
5. Launch per §7.

## 10. Residual risks
- **Retrieval thresholds.** Tier 3 thresholds for retrieval (10 % of P1's Δ_C4 ≈ 3.5–6.8 items) are within numerical jitter, so `pass_c4` on R@k may turn on 1–2 item flips. It is reported as registered.
- **E2 at K8** compares against a neuronic cell.
- **B-F timing.** Co-tenant times for B-F cells have not been measured on this box since August, so the ETA range stays wide until chunks 3–4.
- **NAS stalls.** The NAS (CIFS) can stall; any item substitution is caught by `--expected-stream-count` and the registry.
- **exp_22 dependency.** exp_22's backend amendment and its watcher's reservation support belong to an exp_22 round (decision 7), not to exp_35 code. The exp_22 watcher stays suspended until then.

## 11. r1 → r2: Codex plan r1 (REVISE, 2 BLOCKING + 4 HIGH + 1 NIT) → the BINDING implementation contract
Review: `orbitrir_yaw_evidence_codex_plan_r1_review.md`. Codex: "Yixun does not need to be asked another decision. These fixes implement decisions he already approved." Where this section conflicts with §3b, §5, §6, §7 or §9, THIS section governs.

**B1 — fail-closed import isolation (replaces §3b's "PYTHONPATH set to exactly the overlay" as the mechanism).**
- The overlay becomes ONE module file `overlays/no_flash/flash_attn.py` (same 4-line content as §3b — its sha256 is recomputed and pinned in the launch entry), with NO `__init__.py` anywhere under `overlays/`.
- A committed, same-interpreter launcher `overlays/no_flash/exp35_no_flash_launch.py` runs every exp_35 evaluator AND probe process: it (1) sets `sys.dont_write_bytecode = True` and requires `PYTHONDONTWRITEBYTECODE=1` in its environment; (2) REFUSES if any `__pycache__` / `*.pyc` exists under `overlays/no_flash/`, or if a `flash_attn.py` / `flash_attn/` exists at the repo root or anywhere earlier on the path it builds; (3) builds `sys.path` explicitly = [the overlay dir, the repo root, then the interpreter's own entries with every other occurrence of those two removed] and drops any inherited `PYTHONPATH` entries; (4) asserts `importlib.util.find_spec("flash_attn").origin` realpath == the overlay file; (5) imports `src.models.transformer`, asserts its `__file__` realpath is `<repo>/src/models/transformer.py` and `transformer.flash_attn_func is None`; (6) prints ONE launcher sentinel line with the overlay sha256, its own sha256 and the asserted backend `sdpa`; (7) then executes the target (`eval_FLAC.py` or the probe) with `runpy.run_path(<target>, run_name="__main__")` and the original argv. Any failed assertion exits non-zero BEFORE the target runs.
- The runner spawns `python overlays/no_flash/exp35_no_flash_launch.py eval_FLAC.py <args>` for exp_35 cells (exp_25 cells unchanged); RELEVANT covers both files; their sha256 are runner constants checked at start; the per-cell log check requires the launcher sentinel AND transformer.py's "flash_attn not installed" line, and no "No module named 'flash_attn'"; receipt v2 records `attention_backend: "sdpa"`, the overlay sha256 and the launcher sha256.
- Packaging: `[tool.setuptools.packages.find]` in `pyproject.toml` gains `exclude = ["overlays*"]` (setuptools' namespace discovery would otherwise ship it as `overlays.no_flash`).
- Tests: repo-root shadow (`flash_attn.py` and `flash_attn/`) → refused; a `.pyc` under the overlay → refused; wrong spec origin → refused; inherited PYTHONPATH removed; `flash_attn_func` not None → refused; the sentinel pair required per cell.

**B2 — identity-bound co-tenancy and an atomic hand-over (replaces §7's "ALLOW_SHARED_GPU=1" as the sole gate).**
- Before EVERY spawn, the set of foreign compute processes on the lane's GPU must EQUAL the approved identity tuple, frozen in the launch entry: for GPU 1 (`GPU-6ac2604a-e84e-0615-1221-ccbed7c268da`) exactly Yixun's s43 rank — PID 3028196, its start time (field 22 of `/proc/<pid>/stat`), cwd `/home/yixunhu/codespace/flac-fork-attn`, cmdline sha256, and its log path. Any extra, missing or changed process (PID reuse included) → NO shared spawn → hand-over.
- Hand-over: only when GPU 0 has ZERO foreign compute processes (checked twice, 60 s apart), the supervisor writes the GPU 0 reservation file, then removes the GPU 1 file, and spawns the next cell on GPU 0 with `ALLOW_SHARED_GPU` UNSET. If GPU 0 is not free, the supervisor waits (no spawn) and reports.
- A committed supervisor `exp35_supervisor.py` (round 6a) owns the chunk loop: the frozen 33-chunk order; before each chunk the guard (exit 0), the identity check and the reservation; spawns the runner with `--only <one cell>`; polls the guard every ≤ 5 min while the cell runs (a violation means: no further spawn after the running cell; the runner is never signalled); after chunks 1–2 runs `--exp35-ctl-check` (E1 — mismatch = hard stop), after 3–4 `--exp35-e2` (decides the 45° source only), after 5 validates the probe. Persistent state `exp35_supervisor_state.json` (in the exp_35 folder of the RUN worktree, untracked): the baseline (the time and his step at our first spawn, for the running mean), the chunk ledger, the reservation holder. It never signals any process.
- Tests: trainer exit → hand-over, never another shared spawn; table job replacing the trainer → no shared spawn; an extra PID; PID reuse (same PID, new start time); the reservation transfer (both files never absent together while a cell runs); `ALLOW_SHARED_GPU` unset on GPU 0; guard baseline persistence; log truncation / rotation; epoch boundaries.
- The exp_22 watcher stays ABSENT (not built) through phase (i); the launch entry confirms it.

**H3 — probe provenance.** The B-F probe runs through the same launcher; `orbitrir_bf_cond_probe.json` records `attention_backend`, the overlay and launcher sha256, the H35 campaign digest, its log path and log sha256; the collector's `PROBE_CONTRACT_BF` validates them.

**H5 — packaging:** B1's `pyproject.toml` exclusion.

**H6 — figure provenance fields:** `source_artifact_path`, `source_artifact_sha256`, `source_row_key`, `orbit_execution_basis` (`recorded` | `inferred from the run date after 1479304`) replace §6's single `source_sha256`.

**N7:** every computation and test reads the P1-pin Δ_C4 (and every exp_25 number) at full precision from the hash-pinned exp_25 JSON; the rounded values in §4 are prose only.

**R8 — the critical path for launching phase (i):** rounds 1–4, 5a AND 5b (the Tier 2-C4 / Tier 3 / tables implementation is frozen and reviewed BEFORE any phase-(i) data is observed; rendering included), and round 6a (the guard + the supervisor + identity + hand-over). Round 6b (the off-orbit figure) follows while phase (i) runs, in a separate worktree (the run worktree never moves during a phase). Validation ladder unchanged (§9) plus: a supervisor dry run that plans the 33 chunks and exercises the identity check read-only against the live GPU state without spawning anything.

## 12. r2 → r3: Codex plan r2 (REVISE, 3 BLOCKING + 1 HIGH) → binding additions to §11
Review: `orbitrir_yaw_evidence_codex_plan_r2_review.md`. Where this section conflicts with §11 or earlier sections, THIS section governs.

**Facts read from Yixun's exp_30 queue (read-only, `~/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/exp30_attn_queue.sh`, 2026-11-05):** `TRAIN_JOBS=('ACEattn 44' 'ACEattn 43')` (:239) — s43 is the LAST training; trainings run on GPUs 0,1 (`CUDA_VISIBLE_DEVICES=0,1`, :123); the evaluation FIFO runs ONLY on GPU 1 (`exp30_attn_eval.sh 1 …`, :66); its admission is by free memory with "No exclusivity" (:5-9) — it does NOT honour our reservation. Hence: after s43 ends, his queue puts nothing on GPU 0, and its s43 table job starts on GPU 1.

**B1 — GPU ownership (replaces §11 B2's hand-over paragraph).**
1. The approved identity tuple is enforced TWICE: by the supervisor before each spawn AND by the runner's final pre-exec lane check (the tuple is passed to the runner; any extra / missing / changed foreign process on the lane GPU → the runner refuses that cell, no receipt).
2. The tuple binds the process mechanically: PID, start time (`/proc/<pid>/stat` field 22), cwd (`/proc/<pid>/cwd`), the cmdline sha256 (`/proc/<pid>/cmdline`), and the log path as `readlink /proc/<pid>/fd/1` (and fd/2) — never an asserted string.
3. **End-of-training predictor (prevents a shared cell from overlapping his s43 table job on GPU 1):** before each SHARED spawn the supervisor projects the s43 end from his log — remaining steps (40,000 − current) × 4.208 s/step (the FASTEST unshared 500-step window, so the earliest possible end) — and spawns only if that end is later than now + 2 × the cell's planning duration (vanilla K1 17.6 / K8 19.2 min; B-F K1 24 / K8 35 min). Otherwise it spawns nothing shared and waits for the hand-over. Residual risk (disclosed): none from his queue under this rule unless his training ends earlier than its fastest observed rate allows.
4. **Reservation files are created atomically** (`open(O_CREAT|O_EXCL)`, never overwritten; content: holder exp_35, H35, lane GPU UUID, epoch, created time). The supervisor owns exactly one reservation at any time between its first spawn and its release; the state file records the reservation epoch. Hand-over to GPU 0 happens only when the s43 trainer has exited (its identity gone from both GPUs): (a) GPU 0 has zero foreign compute processes on two checks 60 s apart; (b) the GPU 0 file is created with O_EXCL (if it exists — another holder — refuse and wait); (c) the state records the transfer (epoch + 1) atomically; (d) only then is the GPU 1 file removed. A restart that finds neither, both, or a file not matching the state's epoch → refuse (stop latch) and report.
5. On GPU 0 the runner runs with `ALLOW_SHARED_GPU` UNSET and its own lane gate refuses any foreign process at pre-exec. Residual risk (disclosed): another session's job could start on GPU 0 after our pre-exec check; our cell would then share the card until it ends — the supervisor detects it before the next spawn and stops.

**B2 — the supervisor is a fail-closed state machine (`exp35_supervisor_state.json`, schema v1).** Fields: campaign / H35 / cells sha / supervisor sha; the baseline (time of our first spawn, his step then, his log path + inode + size then); per chunk `planned → in_flight → runner_verified → gates_verified` (with the runner rc, the receipt sha, the gate results); the lane GPU and the reservation epoch; a permanent `stop_latch` {reason, time}. Writes are atomic (temp file + fsync + rename); a corrupt or inconsistent state refuses to run. Restart reconciliation: a chunk with a verified receipt but no `gates_verified` re-runs its gates (E1 / E2 / probe / guard) before anything else; a non-zero runner rc, an absent receipt, a failed verification, a final guard failure or an invalid probe sets the stop latch before any next spawn; a set latch is never cleared by the supervisor (only by the Planner, by hand, after reporting). His log: if its inode changes or its size shrinks → hard stop (no baseline reset).

**B3 — chunk 5 (the probe) transaction.** Chunks 1–4 and 6–33: the runner with ONE manifest cell. Chunk 5: the supervisor invokes the §11 launcher directly on the probe target with the frozen argv, under the same guard, identity, predictor and reservation gates. After the probe process exits and its log is closed, a non-logged finalization step hashes the final log, writes the campaign / backend / overlay / launcher / log provenance into `orbitrir_bf_cond_probe.json`, regenerates its `.sha256`, and runs the collector's `PROBE_CONTRACT_BF` validation; only success marks chunk 5 `gates_verified`.

**H4 — child processes.** (a) The pinned evaluator / probe tree starts no Python child processes (its only subprocesses run `git`); a static test asserts this for `eval_FLAC.py`, the probe and the non-test `src/` modules they import (no `subprocess` / `os.exec*` / `os.spawn*` with a python argv). (b) DataLoader workers: tests force `fork` and `spawn` workers in a toy loader under the launcher and have each worker report `src.models.transformer.__file__` and `flash_attn_func is None`. (c) Defence in depth for any fresh Python child: the runner ALSO exports `PYTHONPATH=<repo>/overlays/no_flash` (only that entry) to the launcher's environment, so a fresh interpreter finds the overlay before site-packages; a test runs a fresh `python -c "import flash_attn"` with that environment from the repo root and asserts the sentinel ImportError.


## 13. r3 → r4: Codex plan r3 (REVISE, 1 BLOCKING + 3 NIT)
Review: `orbitrir_yaw_evidence_codex_plan_r3_review.md` — "The GPU-1 FIFO exclusion above remains the sole blocker once §11 and §12 are faithfully implemented. B2, B3, and H4 no longer block phase (i)." This section governs where it conflicts with §7, §11 or §12.

**B1 — co-tenancy with the s43 table job: the requirement is RESTATED and the residual risk ACCEPTED by the Planner (Codex's option 3).** Yixun's decision 6 is "co-tenancy on GPU 1 for phase (i) only" — sharing GPU 1 with his s43 TRAINING; "the lane never co-tenants with his table job" (§7) was this plan's own stricter wording, not his. Restated: phase (i) shares GPU 1 only with the s43 training by design; a shared cell is STARTED only if the projected earliest end of s43 — (40,000 − current step) × **3.5 s/step** (17 % faster than its fastest observed 500-step window, 4.208) — is later than now + 2 × the cell's planning duration + **30 min**. His queue appends the s43 evaluation only after the training succeeds (`exp30_attn_queue.sh`), and the s44 evaluation FIFO item is durably complete (`logs/queue_20261101T233607_741947.status`: `ACEattn 44 0 …/epoch=8-step=40000.ckpt`; the queue log: `EVAL_DONE ACEattn s44 cells=40 2026-11-04T16:01:40-05:00`; no eval process running at 11:58 EST Nov 5) — so his table job can overlap a shared cell only if the forecast fails: his training reaching 40,000 steps more than 17 % faster than it has ever run, a cell running longer than 2 × its planning duration + 30 min, or a stale step observation (Codex plan r4 NIT). That residual risk is disclosed and accepted: it would slow his table job (not his training) for at most the one in-flight cell. If a new process appears on GPU 1 while our cell runs, the supervisor latches a stop after the running cell (no further shared spawn), records it, and reports to Yixun. The launch entry re-verifies the s44 completion evidence and records the sha256 of his queue script `9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa` (modified vs its checkout HEAD — only `TRAIN_JOBS` differs, per Codex) and evaluator `94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c`.

**NIT 1 — state durability.** After the atomic rename, fsync the state DIRECTORY as well; persist a content anchor of his log at each observation (the byte offset and the sha256 of the last 4 KiB parsed), so a copy-truncate-and-regrow between observations is detected (anchor mismatch → hard stop).

**NIT 2 — chunk 5's ledger fields:** probe process rc, the probe JSON sha256, its `.sha256` sidecar sha256, the final log sha256, and the `PROBE_CONTRACT_BF` result (in place of the runner rc / receipt sha256 of the other chunks).

**NIT 3 — frozen external evidence:** the launch entry records the sha256 of his queue script, his evaluator and his admission helper (`exp30_attn_queue_state.py`), and the s44 completion lines above.

## 14. Implementation amendments (Planner, 2026-11-05 14:36:36 EST; within the approved plan; Codex checks them in code review C3)
Where this section conflicts with §8, §9 or the Coder's readings of §5, THIS section governs. Neither amendment is a new decision for Yixun: A1 implements §1's cell table, §4's E2 rule and §7's ETA line; A2 places a descriptive item that comes after phase (ii).

**A1 — the E2-FAIL contingency block is pre-registered in the manifest.** §1's table registers "(ii), only if E2 FAILS: P1 and B-F at 45°, K1/K8 × s43–46, 16 cells", and §4 says "FAIL → the +16 phase-(ii) cells run", but §8's and §9's "330 rows / orders 230–329" left them out (a counting error in this plan, found by the Coder). The generator and the collector are in the runner's digest set (`depth_canon_eval_run.sh:137`, RELEVANT), and every exp_35 cell must verify under ONE campaign digest at H35 (§5 item 4), so rows added after H35 would split the campaign. Hence:
1. A 17th block `E2-FAIL` (priority 17, orders 330–345) = {P1, B-F} × 45° × K1/K8 × s43–46, with the per-arm protocol of the matching rows (P1 vanilla; B-F fa_invariant with the `_fa_invariant_a4` suffix), dumps and receipt v2 like every exp_35 cell, and the registry checks of the other fixed-angle cells. Names and suffixes stay injective against every exp_25 and exp_35 row.
2. The manifest has 346 rows (exp_35: 116). The exp_25-only render still equals `1073ef64…` byte for byte. §8's checklist records the new full cells.json sha and the sha of the 116 exp_35 rows, in place of "330 rows" and "the 100 exp_35 rows".
3. The block is excluded from every default selection and never appears in the supervisor's frozen 33-chunk phase-(i) order; the supervisor never spawns it. It runs only in phase (ii), on free cards, through an explicit selection the Planner makes at H35 after E2's verdict is FAIL.
4. The collector enforces it fail-closed in `--exp35` mode (E2 computed as in §4):
   - E2 PASS with any E2-FAIL cell present → refuse.
   - E2 FAIL → the 16 cells are required. Without `--partial`, a missing cell → refuse. With `--partial`, it is listed as missing and that method's 45° point falls back to the labelled s42-only point (n 1).
   - E2 FAIL with the block complete → table (b) 45° for FLAC / OrbitRIR is n 5: s42 from the one-pin SDPA cells (P1: the exp_25 C4+45 θ=45° cell at `d6ca307a`, under `external_reference.exp25`, verified offline; B-F: the BF-pin C4 θ=45° cell at H35), plus s43–46 from the E2-FAIL block at H35. Every point carries its provenance (campaign, pin, backend). The P1 pin mix is disclosed, and E1's exact CTL equality between `d6ca307a` and H35 justifies it. This is pooling for a descriptive table, not a Tier 2 / Tier 3 pairing, so §5 item 3 still holds.
   - Tier 2 and Tier 3 are untouched in every case.
5. Tests: the block's rows, order and names; its exclusion from defaults; the three collector cases; Tier 2 / Tier 3 outputs byte-identical across them.

**A2 — E5 lives with the figure round (6b), not in the collector.** E5 (§4) is descriptive, has no verdict and comes after phase (ii); it needs dump relL2 pairs (B-F 135° / 225° vs B-F 45° at s42; CANON 135° / 225° vs CANON 0°) and metric differences. It is implemented as its own script in round 6b's separate worktree (the collector is in the runner's digest set and freezes at H35), reviewed by Codex before it runs, and its output joins the results file as a separately labelled section. R8's freeze (Tier 2-C4, Tier 3 and the tables before any phase-(i) data) is unchanged.

**The Coder's readings of §5 (interim report, 2026-11-05), confirmed:** (1) E2 is the collector's only CSV reader; its result carries the 45° rows, filled only on a PASS, and table (b) takes them from there; Tier 2 / Tier 3 never see the CSV. (2) Table (a) FLAC / CANON and table (c) CANON / P1 / YAW are pointers into `external_reference.exp25`, resolved by the page. (3) E1 must be "equal" for any results, with or without `--partial`. Reading (4), "on an E2 FAIL table (b) shows the s42 points only", is replaced by A1.4.

## 15. Phase (ii) launch (Planner, 2026-11-06 02:59 EST; within the approved plan; PRE-REGISTERED before any phase-(ii) cell; Codex plan + code review)
Implements §1's phase-(ii) row, §7's "phase (ii) runs without `ALLOW_SHARED_GPU`, so only on free cards", exp_22 plan A7 r2 (v)(3) ("exp_35 phase (ii) takes its lanes only under Planner-written O_EXCL reservations (holder `exp35`), written before its runner starts") and the 20:23 launch entry's "phase (ii) re-pins the external CANON manifest / checkpoint at its own launch". Where it conflicts with an earlier section about phase (ii), THIS section governs.

1. **Cells** — exactly the 68 rows of blocks `OFF-P1` (20), `OFF-BF` (20), `OFF-CANON` (20), `CANON-45` (8) of `depth_canon_cells.json` at H35 (sha256 `440eb1befb3445e0b26b01b8e8fa12cb2c37fed76c900864e7efbed5e9e7037d`; 346 rows); the `E2-FAIL` block (16) is NOT selected (E2 PASSED 14 / 14); scored dumps on every cell (no `--no-dump-blocks`), dump root `/media/diskstation/yixunhu/FLAC/exp25_pred_waveforms` (the manifest's).
2. **Pin** — from the run worktree `~/codespace/exp-35-orbitrir-yaw` at H35 `5b065801e9d1c1844461ec84cd0558702c714326`, clean, the campaign digest unchanged (the runner checks both at start, before every spawn and after every cell); no commit there until the phase ends.
3. **CANON re-pin at launch** — `worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256` must hash to `1e4814472b72375d6f3f42b3773f7bdb5d137719f882d00aef0387b770218cba` and carry `52c64b525c8a34f1721b73abd5c4aad67295ea2f729426e9330104b6d1441ace` for `…/exp25_CANON/checkpoints/epoch=8-step=40000.ckpt`; it is passed as `--canon-ckpt-manifest`; the runner stages the checkpoint into its content-addressed cache and re-hashes it before every spawn.
4. **Preconditions** — phase (i) COMPLETE (the supervisor's state 33 / 33 `gates_verified`, no latch, the supervisor process gone, its GPU 1 reservation removed); the Planner's phase-(i) report to Yixun sent (the measured slowdown, the phase-(ii) per-cell time and ETA, the card plan); his s43 training ended.
5. **Lanes** — as Yixun decides on that report; without an answer, ONE lane on GPU 0 (the first free card: exp_35 first, per his order of 2026-11-04 / 2026-11-05), GPU 1 left to the exp_22 pre-flight after his s43 table job; a second exp_35 lane only by a new Planner decision recorded before it starts.
6. **Admission and reservation, per lane card** — no reservation file of any holder; no compute process at two checks 60 s apart (§11 B2's rule); then `/home/yixunhu/codespace/.gpu_reservations/<uuid>.json` created O_EXCL (holder `exp35`, phase `ii`, pin H35, the card's uuid and index, created, the run id), its exact bytes' sha256 and device / inode kept; the card re-checked AFTER the acquisition (a foreign process or file → our own authenticated record released, no launch).
7. **Runner** — first `DRY_RUN=1` of the exact command (it must plan exactly the 68 cells, each with its argv, and refuse nothing); then, detached (`setsid` + `nohup`, the phase-(i) launch environment `env -i HOME USER LOGNAME SHELL LANG PYTHONDONTWRITEBYTECODE=1 PATH`, cwd = the run worktree, stdin `/dev/null`): `EXPECT_HEAD=5b065801e9d1c1844461ec84cd0558702c714326 bash worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh --blocks OFF-P1,OFF-BF,OFF-CANON,CANON-45 --lanes <lane indices> --canon-ckpt-manifest worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256`, NO `ALLOW_SHARED_GPU`, NO `EXP35_APPROVED_COTENANTS`; stdout / stderr to a timestamped log in the run worktree's exp_35 folder (untracked until the phase ends); its identity (pid, start time, cwd, argv sha256) recorded. The runner waits on a busy card and never shares (its own rule).
8. **During** — a read-only watch (the Monitor tool) on the runner's identity, its log (exit / failure lines) and its receipts; never a signal.
9. **After** — on the runner's exit (any rc) the Planner releases each exact reservation record (bytes + inode authenticated; never while a runner of ours lives); rc 0 → the collector's `--exp35` results, the off-orbit figure and E5 (round 6b, from `~/codespace/exp-35-round6b`); rc ≠ 0 → no relaunch without a Planner review and a report to Yixun.
10. **Tooling** — `exp35_phase_ii.sh` in this folder (`reserve` | `dry-run` | `launch` | `status` | `release`), written test-first by an Opus Coder (a sandbox like exp_22's tools; the real runner never started by a test), Codex code review before any use.

**§15 r1 clarifications (Planner, 2026-11-06 03:07 EST; Codex plan §15 r1 = APPROVE, NIT 1–3 folded; its S6 kit contract is BINDING for the Coder and the code review, ):** (a) item 7's "never shares" reads "never intentionally launches on a card observed busy" — the runner checks the lane at selection, right before the spawn and inside the exp_35 child right before the launcher; a foreign job starting after that last check can overlap one in-flight cell (the TOCTOU residual disclosed in §12), and an occupation seen between the parent and the child check makes the campaign exit rc 2 instead of waiting. (b) item 2's "clean" means the runner-relevant tree committed and clean (the worktree necessarily holds untracked phase-(i) evidence); the kit records and compares the exact campaign digest . (c) item 5: the complete lane list is frozen before the single initial runner launch (the runner's global lock admits no second runner); a second lane can only be part of that initial list, by a Planner decision recorded before the launch.
