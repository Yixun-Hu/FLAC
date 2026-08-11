# plan_yaw_aug — exp_15: random-yaw training augmentation for vanilla FLAC vs a matched no-aug control

**Planner:** Claude Fable 5 (main session, neuronic cluster) · **Date:** 2026-08-10 · **Status:** REV 2 (post Codex plan review, verdict REVISE — all findings addressed, §11 changelog) — awaiting Yixun approval. No implementation before approval.

---

## 1. Question and estimand

**Question (Yixun, query 1 + confirmed design decisions):** Does training vanilla-conditioned FLAC on the standard AR train split with a **random horizontal yaw scene shift applied to every training sample** change (a) clean Table-1 performance and (b) yaw robustness, relative to a matched vanilla control trained without the augmentation?

**Arms.**

- **YAWAUG** (new training run): vanilla conditioning (`cond_method` absent ⇒ vanilla; NOT `fa_invariant`), exp_11 rung 8×8 (8×L40 × micro-batch 8 = eff. 64 = SyncBN batch), training seed 42, 40,000 steps, grad-ckpt ON, EMA on — identical to VANL's recipe in every respect except `training.yaw_aug` enabled.
- **VANL** (control, reused per Yixun's decision): exp_11's vanilla arm at step 40,000 (`outputs_FLAC/exp11_VANL/…/epoch=8-step=40000.ckpt`, job 3661520). **Framing (review F6): a *historical recipe-matched control*, not a strictly contemporaneous single-delta control** — it was trained at exp_11's commit `81ddac3`, while YAWAUG trains at the exp_15 pin. §3.3 states what is matched, what is not, and the audits bounding the residual; causal language in the analysis is constrained accordingly.

**Estimand.** For each arm `a ∈ {YAWAUG, VANL}` and context size `K ∈ {1, 8}`, on the full published unseen split (announcement 01: 6,337 items / 17 rooms), EMA weights, cfg 1.0, 1 diffusion step, bf16 conditioning autocast, per-scene-mean aggregation, `--cond-method vanilla` for BOTH arms (both are vanilla-conditioned models; announcement 05):

- `m_Z(a, K)` — **Table-1 block (headline)**: unrotated (θ = 0) metrics, 5 eval seeds 42–46. The comparison Yixun literally asked for.
- `m_R(a, K)` — **robustness block (confirmed secondary)**: every eval sample independently draws a random yaw `θ_i = d_i · (360/512)°`, `d_i ~ Uniform{0,…,511}` (exact column roll, no interpolation), applied as the physically-consistent rotation (depth roll + per-pixel 3D vectors + all four pose fields together; `context_audio` and GT RIR untouched). **Protocol identical to exp_14's R block — exp_15 consumes exp_14's eval machinery and assignment-integrity contract verbatim (§6.1).**
- Readouts: clean contrast `m_Z(YAWAUG) − m_Z(VANL)`, paired degradation `Δ(a) = m_R(a) − m_Z(a)`, absolute `m_R` ranking.

## 2. Why this experiment (context)

- exp_02: vanilla FLAC is **not** yaw-invariant (Metric-1 rel-L2 0.19–0.22 under C₄ panorama rotation).
- exp_11: trained the architectural alternative (C4/C8/C16/C32 frame-averaged arms); θ=0 headline metrics *degrade* with orbit size at matched steps.
- exp_14 (in implementation, concurrent session): measures trained-arm robustness under per-sample random yaw; owns the eval-side random-yaw machinery (its plan §9 records the seam split with exp_15 explicitly).
- **Missing arm:** the classical data-side alternative — *augmentation instead of architecture*. A rigid yaw rotation of the scene (listener panorama + all source/context poses about the listener's vertical axis) leaves the true RIR unchanged, so rotated conditioning + unchanged GT is a physically-valid training pair for free, with zero inference-time cost (contrast FA's n× conditioner passes). No such model exists in the program.
- Honest outcome space: augmentation may regularize (clean metrics improve), be neutral, or cost clean performance while buying robustness — and it may also *fail* to buy robustness (a reportable negative, not a forbidden one; §5).

## 3. Training design

### 3.1 Augmentation protocol (pre-registered, per Yixun's confirmed choices)

- **Fresh draw per visit:** each time a sample is loaded during training, a new independent column offset `d ∈ {0,…,511}` uniform. Over ~8.8 epochs (291,210 items, 4,550 steps/epoch at eff. 64) each item is seen with ~9 different yaws.
- **Physically consistent, exact:** apply `rotate_scene_metadata(md, d·2π/512, img_w=512)` — the existing exp_02 utility: equirectangular depth (shape **3×256×512**) rolled by exactly `d` columns, its per-pixel 3D vectors and all four pose fields (`source`, `source_vit`, `context_poses`, `context_poses_vit`) rotated about z by the same effective angle. `context_audio`, `padding_mask`, `scene`, and the GT RIR (`reals`) pass through untouched.
- **Applied in `training_step` ONLY.** `validation_step`/`test_step` code paths are untouched (no augmentation branch at all). The arm trains with `--val-every -1` (exp_11 rung convention).
- **Resume-exact, counter-based draws (review F5).** Offsets are a pure function of `(yaw_aug.seed, global_step, global_rank, within-batch index)`: at each training step, a fresh `torch.Generator` is seeded with a SplitMix64-style mix of `(yaw_aug.seed, global_step, global_rank)` and draws the micro-batch's offsets. There is no stream state to checkpoint, so an interrupted run resumed at step N reproduces exactly the draws an uninterrupted run would have made — the "fresh draw per visit" policy survives restarts without replaying a prefix. World size and micro-batch are pinned across any RESTART leg by the kit's rung gate (§6.6), so the (rank, step) → sample mapping is stable. Global Python/NumPy/torch RNG streams are never touched by generator construction, drawing, or application (tested §6.5) — the YAWAUG run differs from VANL only through the augmentation itself, not through RNG displacement.
- **Runtime schema guards (review F7), fail-closed ValueError at the hook:** metadata is a nonempty list of dicts; every sample carries `depth` with shape `[3, H, W]` and `W == img_w == 512` (the roll width is validated against the actual tensor, never trusted from config alone); pose fields present with trailing dimension 3; dtype/device preserved by the rotation (tested). `enabled` must be literal `true`, `img_w` a positive int, `seed` an int.

### 3.2 Recipe pins (identical to exp_11 rung 8×8)

torchrun 1 node × 8 L40 · `--batch-size 8 --accum-batches 1` (eff. 64 = SyncBN batch; `--sync-batchnorm true`) · `--seed 42 --num-workers 6 --precision bf16-mixed --val-every -1 --gradient-clip-val 0.0 --checkpoint-every 2500 --max-steps 40000` · ViT `gradient_checkpointing: true` · EMA on · AdamW 5e-5/(0.9, 0.999)/wd 1e-3 · InverseLR(1e6, 0.5, 0.99) · dataset config `src/configs/dataset_configs/AR/train/acousticroom_train.json` · VAE `weights/FLAC/VAE.safetensors` (sha `8d82159e…`).

**Arm config:** `exp_15…/FLAC_AR_YAWAUG.json` = byte-copy of exp_11's `FLAC_AR_VANCKPT.json` (sha `733ca52b…` = VANL's registry-pinned config) + exactly one addition: `"yaw_aug": {"enabled": true, "img_w": 512, "seed": 42}` in the `training` block. A pytest asserts the diff is exactly that block (§6.5-8).

### 3.3 Matched-control statement and audits (review F2, F6)

**Matched:** cluster + GPU model, rung topology (8×8, SyncBN 64), training seed, step budget, checkpoint cadence, dataset config, VAE (sha-gated), optimizer/schedule, EMA, grad-ckpt, env versions (torch/PL gates). **Not matched:** code state (VANL at `81ddac3`, YAWAUG at the exp_15 pin) and wall-clock era. Audits bounding the residual:

1. **Control-admission record (F2 — BLOCKING fix).** exp_11's registry pins VANL's launch manifest/commit/config/VAE/rung/seed but **not** the 40k checkpoint itself (no `final_ckpt_sha256` — the job ended `FAILED` post-save and the field was never backfilled). exp_15 therefore creates its own committed, immutable `yaw_aug_control_admission.json` (exp_11's files untouched): canonical checkpoint path; sha256 of the checkpoint file; embedded `global_step == 40000`; embedded config hash vs `FLAC_AR_VANCKPT.json`; presence/readability of the EMA state; cross-references to exp_11's manifest sha `113d06a2…`, commit `81ddac3`, training seed 42, rung 8×8, VAE sha. Written once by a reviewed recorder script, then read-only; **every VANL eval cell re-hashes the checkpoint and validates this record before running (fail-closed).** YAWAUG's checkpoint is admitted the same way via `yaw_aug_launch_registry.json` (§6.6).
2. **Launch-pin diff allowlist (F6).** The training sbatch gate computes `git diff --name-only 81ddac3..<pin>` restricted to production training surfaces (`train.py`, `defaults.ini`, `src/`) and asserts every changed file is on the committed, reviewed allowlist (`yaw_aug_pin_allowlist.txt` — expected: `src/training/factory.py`, `src/training/diffusion.py`, `src/data/yaw_rotation.py`, `eval_FLAC.py` [exp_14's], `src/tests/*`). Any surprise file ⇒ abort.
3. **Golden disabled-path regression (F6).** BEFORE any exp_15 code lands, a fixture-capture harness records, at the pre-change commit, a deterministic reference for a seeded synthetic batch: conditioning outputs, loss value, and global RNG state snapshots through a `training_step`. The post-change test replays it with `yaw_aug` absent and asserts equality. This upgrades "the disabled path is untouched" from object-identity to whole-step behavioral evidence. (Terminology per F12: the claim is *behavioral equivalence under the disabled-path regression*, reserved-word "byte-identical" only for serialized artifacts under snapshot tests.)
4. **Factory hygiene (F6):** when `training.yaw_aug` is absent, the factory passes **no new kwargs at all** — the wrapper construction call is literally the pre-change call.
5. **Env cross-check (F6):** the kit's env gates (torch/PL versions, VAE sha) are additionally compared against the values recorded in VANL's launch manifest.

W&B curves are context only — the treatment intentionally changes conditioning and loss, so curve differences are expected and diagnose nothing about the no-op path (F6).

## 4. Evaluation design

### 4.1 Cells (42 single-GPU jobs, one campaign pin)

| Block | Cells | Protocol | Count |
|---|---|---|---|
| **T** (Table-1, headline) | 2 arms × K∈{1,8} × eval seeds 42–46 | θ = 0, fixed mode | 20 |
| **R** (robustness, secondary) | 2 arms × K∈{1,8} × eval seeds 42–46 | per-sample random yaw, rotation seed = eval seed | 20 |
| **V** (controls, s42 K=8 only) | VANL @ fixed 90° (positive control, gate G1) · YAWAUG @ fixed 90° (descriptive/mechanistic ONLY — review F3: no gate role, since gating on it would presuppose the augmentation succeeded) | fixed 90° | 2 |

- All 42 cells at **one** campaign pin. VANL T-cells re-run at our pin; exp_11's committed VANL rows and exp_14's Z rows (when they land) are **non-halting external reproduction checks** (§5, review F3), never the paired reference.
- R cells for both arms at the same (K, seed) share the identical per-item rotation assignment (rotation seed = eval seed, dataloader stream order) ⇒ cross-arm contrast is rotation-matched — **verified by exp_14's assignment-integrity hashes, not assumed** (§4.3).
- Announcement 01: the rotation perturbs conditioning of the existing full split; the item set is never altered. No new eval configurations.

### 4.2 Pinned literal argv per cell class (review F9 — flags, not just manifest fields)

Defaults are unsafe (`--frame-avg-angles` defaults to the C4 list, `--cond-autocast` defaults to `default` ≠ bf16), so every invocation pins the full protocol explicitly; the kit's manifest records it and the collector re-validates it per cell class before aggregation:

- **T:** `--cond-method vanilla --frame-avg-angles 0,90,180,270 --cond-autocast bf16 --rotate-mode fixed --rotate-deg 0`
- **R:** `--cond-method vanilla --frame-avg-angles 0,90,180,270 --cond-autocast bf16 --rotate-mode random --rotate-seed <eval-seed> --rotate-deg 0`
- **V:** `--cond-method vanilla --frame-avg-angles 0,90,180,270 --cond-autocast bf16 --rotate-mode fixed --rotate-deg 90`

(`--frame-avg-angles` is inert under vanilla but pinned per announcement 05's explicit-flags rule. `--rotate-seed` is never passed in fixed mode — exp_14's guard makes that a hard error.) Eval `batch_size`/`num_workers` are campaign constants (exp_14 §3.2 pins, same values), recorded per manifest; deviating cells are rejected by the collector.

### 4.3 Assignment integrity (adopted verbatim from exp_14 §3.3 — review F1)

exp_15 adopts exp_14's full contract, not an offset-only hash (the dataset can recursively substitute failed items and context selection is stochastic, so offset-stream identity alone proves nothing about what was actually evaluated): per-position `(i, target_relpath, context_ids, img_w[, d_i])` tuples with canonical JSON serialization; **`input_hash`** and (R/V) **`assignment_hash`**; exactly **6,337** positions with the substitution guard (target relpath must match the split-manifest order — cell FAILS otherwise); sidecar records both hashes + tuple count; `_rotrand<seed>` naming token. Collector equality checks, fail-closed: across arms within (K, seed) — `input_hash` equal for both arms and `assignment_hash` equal for R cells; within (arm, K, seed) — `Z.input_hash == R.input_hash`. Any inequality ⇒ the affected contrast renders BLOCKED, never a number.

## 5. Hypotheses, statistics, decision rules (pre-registered; review F3, F4)

**Estimation conventions (all contrasts):** the per-seed observation is the per-scene-mean aggregate for that (arm, K, seed) cell; contrasts use the 5 seed-paired differences; estimate = mean difference; two-sided 95% paired-t CI, df = 4, α = 0.05. **Metric directions (lower = better unless stated):** T60% ↓, C50 ↓, EDT ↓, FD ↓, R@1 ↑, R@5 ↑, R@10 ↑. Co-primary metrics **T60%** and **R@1**. K=8 confirmatory; K=1 repeats everything descriptively.

**Multiplicity (one confirmatory family):** the confirmatory family is **H1's two co-primaries only** (Holm over 2 tests). H2 and H3 are secondary: reported with the same paired machinery and CIs, explicitly labeled secondary, no confirmatory claims. All remaining metrics everywhere: descriptive with unadjusted CIs.

- **H1 (PRIMARY — clean cost/benefit, the literal ask):** `m_Z(YAWAUG) vs m_Z(VANL)`, K=8, T60% and R@1, two-sided (no directional prior), Holm-2. Per-metric verdicts: **YAWAUG-SUPERIOR** / **YAWAUG-INFERIOR** (Holm-adjusted p < 0.05, direction per the metric's orientation) / **NOT STATISTICALLY RESOLVED** (no equivalence claims — "EQUIV" is retired for this experiment; review F4).
- **H2 (secondary — augmentation buys flatness):** per-seed oriented degradation `δ_m(a, s) = orient(m_R(a,s) − m_Z(a,s))` (orientation flips sign for ↑-metrics so positive = worse). Contrast `d_s = δ(VANL, s) − δ(YAWAUG, s)`, expected > 0; paired-t CI on T60% and R@1. **A negative or null H2 is a reportable scientific outcome.** Triage on failure is neutral and pre-specified (review F3): the G2/G3 evidence, unit/integration tests, and V-cell readouts are examined and reported; "implementation defect" and "augmentation genuinely fails to confer robustness" both remain on the table until that evidence discriminates them.
- **H3 (secondary — deployment):** absolute `m_R(YAWAUG) vs m_R(VANL)`, K=8, seed-paired CIs on the co-primaries; all metrics descriptive.
- **Cross-experiment (descriptive, no inference):** YAWAUG placed alongside exp_14's VANL/C4L/C8/C16/C32 robustness rows when those land (augmentation-vs-architecture). Cross-pin status disclosed; the shared rotation-seed scheme makes R assignments nominally identical across campaigns — checked via `assignment_hash` comparison, disclosed either way, never assumed.

**Scope-of-inference statement (review F4, mandatory in `_results.md` and `_analysis.md`):** both arms have exactly one training run each (seed 42). Five eval seeds estimate evaluation-time variability (diffusion sampling + rotation assignment), NOT training-run variability (init, data order, hardware nondeterminism) and NOT checkpoint-band position — matching seed and step aligns the two draws' schedules but cannot pair away band variance. All inference is conditional on these two specific training trajectories at the pre-registered 40k endpoint (no checkpoint selection); the checkpoint-band caveat (HANDOFF standing lesson) is repeated verbatim in the analysis.

**Validity gates — all executable harness checks (review F3); H-readouts are read only after they pass:**

- **G1 (positive control):** `m_T60(VANL V@90°, s42, K8) − m_T60(VANL T, s42, K8) ≥ 5 · σ̂_T60(VANL)`, where `σ̂_T60(VANL)` = std over VANL's 5 T-cell seeds at K=8 (T block completes first). exp_02's ~3.4 pp prior sits far above this bound. Failure ⇒ HALT (the harness is not detecting non-invariance).
- **G2 (golden assignment):** the smoke R-cell's sidecar offset sequence for rotation seed 42 equals the sequence pre-computed in the unit test (proves draws reach `rotate_scene_metadata`; backed by a pytest integration spy — exp_14 G3 verbatim).
- **G3 (assignment integrity):** every §4.3 hash equality holds; violations render the affected contrast BLOCKED.
- **G4 (checkpoint admission):** every cell re-hashes its checkpoint against `yaw_aug_control_admission.json` (VANL) or `yaw_aug_launch_registry.json` (YAWAUG); mismatch ⇒ HALT.
- **G5 (completeness):** 5/5 seeds per (arm, K, block) before any mean ± std is emitted; partial cells render PENDING, never numbers.
- **External checks (non-halting, pre-declared formula):** VANL T rows vs exp_11's committed VANL rows (if/where populated) and vs exp_14's Z rows when they land: mean differences reported; discrepancies beyond `3·√(σ_a² + σ_b²)/√5` disclosed in the analysis (exp_14 G5 formula). Cross-pin, does not halt — an honestly measured discrepancy does not invalidate the within-pin exp_15 comparison (review F3).
- **YAWAUG V@90° carries no gate role** — it is reported in the mechanism section only.

## 6. Implementation plan (per file)

Role split per SOP: Opus 5 max-effort Coder implements; Codex `gpt-5.6-sol` xhigh reviews per round; TDD (red→green, one small commit per cycle) for every new function; tests in `src/tests/`. Commits < 200 lines, base = current HEAD of `check-equivariance-necessity` (pull-rebased at implementation start; base SHA in `commits_yaw_aug.md`).

### 6.1 Hard dependency: exp_14 owns the eval seam (review F1 — BLOCKING fix)

**exp_15 makes ZERO edits to `eval_FLAC.py` and to exp_14's helpers in `src/data/yaw_rotation.py`.** Per exp_14's plan §9 (the recorded seam split both sessions see): exp_14 owns `--rotate-mode random`/`--rotate-seed`, the assignment-integrity machinery, and `draw_yaw_offsets`/`offsets_to_radians`; exp_15 owns the training-side hook and **builds on exp_14's committed, review-closed rounds** (round 1 has already landed: `9e737a1` helpers, `66e6ca5` fixed-mode byte-compat contract). Sequencing: exp_15's training-side work (§6.2–6.3) and kit prep can proceed immediately — they touch disjoint files — but **exp_15's eval cells cannot launch until exp_14's eval rounds are review-closed at a pin exp_15 can descend from.** If exp_14 stalls, exp_15 escalates to Yixun rather than forking the spec. If exp_14 amends its context-identity fingerprint (the known open issue that AR metadata does not currently expose context-source IDs), exp_15 adopts the amended contract as-is.

### 6.2 `src/training/factory.py` — extend (≈ +15 lines)

- Parse optional `training.yaw_aug`: `{"enabled": bool, "img_w": int, "seed": int}`. **Absent block ⇒ no new kwargs passed — the construction call is the pre-change call** (F6).
- Fail-closed guards (ValueError): `enabled: true` with `cond_method == "fa_invariant"` (untested combination, out of scope); `enabled: true` with missing `seed` or `img_w`; unknown keys in the block; non-literal-boolean `enabled`.

### 6.3 `src/training/diffusion.py` — extend (≈ +40 lines)

- `DiffusionCondTrainingWrapper.__init__` gains `yaw_aug_enabled: bool = False`, `yaw_aug_img_w: int = 512`, `yaw_aug_seed: int = 0`. Rank-0 prints `yaw_aug ENABLED img_w=512 seed=42` once at fit start (launch log gate greps for it — evidence a branch was entered; the *correctness* evidence is the §6.5 test suite, per review).
- `training_step` only, before `_compute_conditioning`: if enabled — run the §3.1 schema guards; seed a fresh generator from SplitMix64-mix `(yaw_aug_seed, global_step, global_rank)`; `offsets = draw_yaw_offsets(len(metadata), img_w, gen)` (exp_14's committed helper); `metadata = [rotate_scene_metadata(md, a, img_w) for md, a in zip(metadata, offsets_to_radians(offsets, img_w))]`. `reals` untouched.
- `validation_step`/`test_step`: untouched code paths.

### 6.4 Control-admission recorder — `yaw_aug_record_control.py` (exp_15 folder, ≈ 60 lines, TDD)

Writes `yaw_aug_control_admission.json` per §3.3-1 (checkpoint sha256, embedded step/config checks, EMA presence, exp_11 cross-refs). Copy-only/read-only with respect to the checkpoint; refuses to overwrite an existing record. Test list: §6.5-9.

### 6.5 `src/tests/test_yaw_aug_training.py` — new (TDD, written first)

1. Counter-based determinism: draws for (seed, step, rank) reproduce; **uninterrupted steps 0…N+M == draws under simulated resume at N** (F5); distinct steps and distinct ranks decorrelate.
2. Global-RNG isolation: python/NumPy/torch global states are bit-identical before/after generator construction + drawing + augmentation application (F5).
3. Exactness: every drawn angle re-quantizes through `rotate_scene_metadata` with zero error (`dj` round-trips to `d`).
4. Factory parsing: absent block ⇒ construction kwargs literally identical to pre-change reference; enabled block ⇒ flags through; each §6.2 guard raises.
5. **Golden disabled-path regression (F6):** pre-change-captured fixture (conditioning outputs + loss + RNG snapshots on a seeded synthetic batch) replays identically with `yaw_aug` absent.
6. Aug application (stub conditioner capturing metadata): enabled `training_step` passes exactly the per-sample rotation under the same draws (tensor equality vs manual reference); `reals`/`context_audio`/`padding_mask`/`scene` bit-identical; dtype/device preserved; **fixed-offset integration cases d ∈ {0, 1, 128, 511}** verifying depth roll and all four pose rotations with no input mutation (F7).
7. Schema guards (F7): empty metadata, missing depth, wrong depth width (≠512), wrong pose trailing dim, non-bool `enabled` — each raises; width validated against the actual tensor.
8. `validation_step` never augments even when enabled.
9. Config-diff assert: `FLAC_AR_YAWAUG.json` vs exp_11 `FLAC_AR_VANCKPT.json` differ in exactly `training.yaw_aug`.
10. Recorder (§6.4): record content correct on a tiny synthetic checkpoint; refuses overwrite; detects step/config mismatch.

### 6.6 Launch kit — `yaw_aug_train.sbatch`, `yaw_aug_submit.sh`, `yaw_aug_train_guardtests.sh`

- **Commit 1: verbatim copies** of exp_11's kit (review diffs isolate our changes; exp_11's files never touched).
- **Commit 2+ (deltas — enumerated per review F11, since the source kit now encodes exp_11's Q10 extension era):** `PINNED_MAXSTEPS` **100000 → 40000** and the production max-step assertion (`fa_orbit_train.sbatch:189`) re-pinned to 40000; INITIAL single-segment semantics with RESTART retained but **capped at the pre-registered 40k endpoint** (preflight rejects any resume targeting beyond it); time pin 24:00:00 (VANL measured 10.6 h; augmentation overhead benchmarked at rung §7-5); legal arm set = {YAWAUG} only; namespace `exp15_` throughout (`outputs_FLAC/exp15_YAWAUG`); registry `yaw_aug_launch_registry.json` with exp_11's INITIAL + producer-record schema **including `final_ckpt_sha256`** (the omission that caused F2 is not repeated); semantic config gate asserts no `cond_method`, `yaw_aug == {"enabled": true, "img_w": 512, "seed": 42}` literally, grad-ckpt true, EMA on; **post-launch log gate** requires the `yaw_aug ENABLED` line before step 0 (fail-closed kill); **launch-pin diff allowlist gate** (§3.3-2); env cross-check vs VANL's manifest (§3.3-5); argv parity check retained vs the exp_07 reference. Guardtests prove: production launch with MAXSTEPS ≠ 40000 dies; RESTART beyond 40k dies; allowlist gate fires on a planted surprise file.

### 6.7 Eval kit — `yaw_aug_screen.sbatch`, `yaw_aug_screen_submit.sh`, `yaw_aug_screen_guardtests.sh`, `yaw_aug_submit_grid.sh`

Commit-1-verbatim / commit-2-delta from exp_11's screen kit, adapted to consume **exp_14's landed eval machinery** (§6.1): namespace `exp15_`; arms {YAWAUG, VANL}; cell classes {tbl, rrob, vctl} with the §4.2 pinned argv; eval names `exp15_<ARM>_<CELL>[_rot90|_rotrand<seed>]_S40000_s<SEED>_K<K>`; checkpoint admission per §5-G4; leaf-asset provisioning includes `weights/FLAC/VAE.ckpt` (exp_11-R3 root cause, regression-checked in the smoke cell); eval batch/worker campaign pins recorded per manifest. Grid submitter: bounded concurrency ≤ 16, **validate-before-skip dedup** (a cell is skipped only if its existing artifacts pass validation — exp_14 B6 lesson), `DRYRUN=1` printing the exact 42-cell grid, every submission appended to `yaw_aug_command.md` at launch. Guardtests: exact-grid assertion; unregistered cells rejected.

### 6.8 `yaw_aug_collect.py` — new (≈ 150 lines; TDD in `src/tests/test_yaw_aug_collect.py`, per-function inventory per review F8)

| Function | Red tests |
|---|---|
| `parse_cell(path)` | valid/malformed metrics JSON + sidecar; missing fields named |
| `validate_protocol(record, cell_class)` | wrong cond_method / autocast / rotate flags per class rejected |
| `verify_hashes(records)` | cross-arm `input_hash`/`assignment_hash` and Z↔R equality violations detected and named (BLOCKED, not numbers) |
| `check_completeness(cells)` | 4/5 seeds ⇒ PENDING; exactly the 42-cell grid recognized; extra/unknown cells rejected |
| `pair_seeds(a, b)` | seed alignment; missing-seed asymmetry rejected |
| `orient_metric(name, values)` | direction table applied; unknown metric rejected |
| `paired_t_ci(diffs)` | df=4 CI against precomputed reference values |
| `holm(pvals)` | reference Holm ordering on synthetic p-values |
| `gate_report(cells)` | G1 formula on synthetic values (pass + fail); external-check formula |
| `render_tables(results)` | golden markdown fixture |

Output: `_results.md` tables (H1 clean contrast, paired Δ, absolute R, gate report, external checks) + JSON bundle for the HTML. Shared verification helpers are imported from exp_14's collector where it exposes them (Coder's call, reviewed) — never re-implemented divergently.

### 6.9 `model_comparison.md` integration (review F10 — pre-registered transaction, not post-results)

**Trigger:** when the YAWAUG T block reaches 5/5 seeds at both K and the §5 gates for those cells pass, the tested YAWAUG row spec lands in `gen_model_comparison.py` → regenerate → commit → push **immediately** (announcement 04). **exp_15 adds NO VANL row**: the VANL model row is owned by exp_14's §5.7 contract (which replaces the never-populated exp_11 Q9 spec); exp_15's fresh VANL cells remain a results-local reproduction/control block in `yaw_aug_results.md` only.

### 6.10 Post-results (per SOP)

`yaw_aug_params_set_up.md` + `yaw_aug_command.md` at launch; `yaw_aug_results.md`; `yaw_aug_analysis.md` (Planner, incl. the §5 scope-of-inference statement); `yaw_aug_01_results.html` + `yaw_aug_results_assets/` (dataviz guidance loaded first); `commits_yaw_aug.md`.

## 7. Validation ladder (cheapest-first, each rung logged in `_worklog.md`; review F8 restored the SOP's missing rungs)

1. **Static:** `py_compile` on changed `.py`; `bash -n` on the kit; `git diff --check`; `json.load` on `FLAC_AR_YAWAUG.json` + admission/registry JSONs.
2. **Pytest:** new tests (§6.5, §6.8) + existing yaw/conditioning/factory subsets + the disabled-path regression subset green.
3. **Tiny synthetic forward (SOP rung 2):** real `DiffusionCondTrainingWrapper` on a seeded synthetic batch, CPU or single GPU — one `training_step` with aug on and off; asserts the §6.5-5/6 invariants end-to-end in the real class, no full weights or data.
4. **Small real-data readback (SOP rung 3):** a few real AR train records through the actual dataloader path; assert depth is `[3, 256, 512]` float with finite values, pose fields `[…, 3]`, and rotation invariants on real samples (roll-by-d equivalence, pose-norm preservation, z unchanged).
5. **Kit guardtests + `DRYRUN=1`:** train kit (argv parity, gates incl. the 40k re-pin and allowlist) and eval grid submitter (exact 42-cell list, zero submissions).
6. **Training smoke (`SMOKE=1`):** ~20 steps at the real rung on 8×L40, storage-light; acceptance: `yaw_aug ENABLED` line present, ≥1 optimizer step, no OOM/NaN, peak VRAM ≈ rung profile (~9.4 GB checkpointed), and **measured step rate on the actual 3×256×512 augmented path ≥ 0.9× VANL's ~1.05 steps/s** (F7: the cost model is benchmarked, not assumed; a larger drop triages before launch).
7. **Full 40k launch** — only after 1–6 pass, from a pushed pinned commit, §8 acceptance criteria logged first.
8. **Eval (blocked on exp_14 per §6.1):** first `vctl` cell alone (VANL@90°, exercises provisioning incl. VAE.ckpt leaf + fixed rotation + metrics landing); one `rrob` probe (YAWAUG K=8 s42) for timing + gate G2; then full waves (T block, then R block), validate-before-skip dedup.

## 8. Launch acceptance criteria (written to `_worklog.md` at launch time, judged against — not vibes)

Training: job reports the pinned commit SHA; allowlist gate green; 1 node × 8 L40, micro 8, eff 64, SyncBN active; grad-ckpt gate passed; `yaw_aug ENABLED img_w=512 seed=42` logged before step 0; ≥1 optimizer step, no OOM/NaN; step rate within the §7-6 bound; checkpoints on the 2500 cadence; `step=40000` checkpoint written, sha-recorded (with `final_ckpt_sha256`) in `yaw_aug_launch_registry.json`. Eval cells: §4.2 argv verified in the manifest; metrics JSON + sidecar (hashes + tuple count) land; §5 gates pass.

## 9. Risks / standing-rule compliance

- **Concurrent writers + exp_14 dependency.** `git pull --rebase` before every commit; after submitting pin-bound jobs, no tracked-file changes until every job passes its start gate (exp_11 standing rule). exp_14 owns `eval_FLAC.py` + shared helpers (§6.1) — exp_15 never edits them and never edits exp_14's or exp_11's folders. Before each Coder round: check `git log` + exp_14's worklog for its round state.
- **Protocol-flag trap (announcement 05):** §4.2 pins the literal argv per cell class; manifest + collector re-validate. No fa checkpoints touched.
- **Legacy behavior frozen:** `training.yaw_aug` absent ⇒ behaviorally identical under the disabled-path golden regression (§3.3-3, §6.5-4/5); eval fixed mode is exp_14's snapshot-frozen contract. No other experiment's tooling can be perturbed.
- **Historical control (accepted risk, Yixun's call):** §3.3 audits bound it; analysis language constrained (F6); the §5 scope statement is mandatory.
- **Checkpoint-band caveat:** pre-registered 40k endpoint, no selection; band variance is not paired away (§5 scope statement).
- **Storage:** 16 checkpoints × 724 MB ≈ 11.6 GB under `outputs_FLAC/exp15_YAWAUG` (gitignored) + KB-scale metrics; df floor gate in the kit.
- **Slurm etiquette:** one 8×L40 single-node job (~11 h, 24 h pin, partition `all`), then ≤ 16 concurrent single-GPU cells; login node used only for submission and bounded bookkeeping.
- **Codex sandbox:** every review prompt explicitly forbids installing or modifying environments (`-s read-only` is not sufficient protection).

## 10. Cost and schedule

- **Training:** 1 run × ~11 h service time on 8×L40 (~88 GPU-h; 24 h pin). No fresh control (Yixun's decision). Queue wait excluded (service-time label, exp_14 N10 convention).
- **Eval:** 42 cells at measured per-cell times (K=1 ≈ 13 min, K=8 ≈ 23 min, 1×L40) ≈ **13–16 GPU-h**; ≈ 2–3 h service time at ≤ 16 concurrent; **start gated on exp_14's eval rounds closing (§6.1)** — training and eval-kit prep proceed in parallel with exp_14, so exp_14 is unlikely to be the critical path (its round 1 has already landed).
- **Rounds:** Coder + per-round Codex reviews realistically 2–4 h before the training launch.
- **End-to-end:** ≈ 1.5 days from approval to `_results.md`, dominated by the training run.

## 11. Rev 2 changelog (review finding → disposition)

| # | Finding | Disposition |
|---|---|---|
| 1 | BLOCKING — shared-contract adoption + ownership conflict | §6.1 hard dependency: exp_14 sole owner, exp_15 zero edits to the eval seam, builds on landed commits (`9e737a1`, `66e6ca5`); §4.3 adopts the full hash/substitution/tuple-count contract + `_rotrand<seed>`; escalation (not forking) if exp_14 stalls; context-ID fingerprint follows exp_14's resolution |
| 2 | BLOCKING — VANL checkpoint not sha-pinned | §3.3-1 committed `yaw_aug_control_admission.json` (recorder §6.4, tests §6.5-10); per-cell re-hash gate G4; own registry records `final_ckpt_sha256` (§6.6) |
| 3 | BLOCKING — outcome used as gate | YAWAUG@90° demoted to descriptive/mechanistic (§4.1, §5); gates reduced to executable checks G1–G5 with pre-registered formulas; VANL@90° positive control with numeric threshold (G1); cross-pin reproduction demoted to non-halting external check with exp_14's G5 formula; neutral H2-failure triage (§5) |
| 4 | MAJOR — statistics incomplete | §5 conventions block (per-seed observation, directions, df=4 paired-t, α); one confirmatory family (H1 co-primaries, Holm-2), H2/H3 secondary; per-seed `δ` definition; EQUIV retired → NOT STATISTICALLY RESOLVED; mandatory scope-of-inference statement |
| 5 | MAJOR — resume violates fresh-draw / determinism unsupported | §3.1 counter-based stateless draws keyed (seed, step, rank, index); resume-exactness + N+M test (§6.5-1); global-RNG isolation test (§6.5-2); world-size/micro-batch pinned across legs |
| 6 | MAJOR — no-op audit too weak | §3.3-2 pin diff allowlist gate; §3.3-3 golden disabled-path fixture; §3.3-4 factory kwarg omission; §3.3-5 env cross-check; W&B curves demoted to context; "historical recipe-matched control" framing |
| 7 | MAJOR — runtime/schema guards + wrong cost model | §3.1 fail-closed schema guards (width validated against the tensor); depth corrected to 3×256×512; §7-6 benchmarks the real path; fixed-offset integration tests {0,1,128,511} (§6.5-6) |
| 8 | MAJOR — TDD/ladder gaps | §6.8 per-function collector test inventory; ladder rungs 3–4 restored (tiny synthetic forward, real-data readback); exact-grid + reject-unregistered guardtests (§6.7) |
| 9 | MAJOR — flags not pinned | §4.2 literal argv per cell class incl. `--frame-avg-angles`/`--cond-autocast`; batch/worker campaign pins; collector re-validation |
| 10 | MAJOR — announcement-04 timing / VANL row ownership | §6.9 pre-registered immediate-transaction trigger; no exp_15 VANL row (exp_14 §5.7 owns it); exp_15 VANL cells results-local only |
| 11 | MAJOR — kit encodes 100k era | §6.6 enumerated deltas: 40k re-pin + assertion, RESTART cap, time pins, arm set, registry schema; guardtests for both failure modes |
| 12 | MINOR — bookkeeping/terminology | exp_15 row added to `master_experiment_tracker.md` (PLANNED); "byte-identical" reserved for snapshot-covered artifacts, otherwise "behaviorally identical under the disabled-path regression" |
