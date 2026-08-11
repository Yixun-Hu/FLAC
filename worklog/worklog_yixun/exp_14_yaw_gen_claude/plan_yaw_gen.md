# plan_yaw_gen — exp_14: random-yaw generalization of vanilla vs C4/C8/C16/C32 frame-averaged conditioning

**Planner:** Claude Fable 5 (main session, neuronic cluster) · **Date:** 2026-08-10 · **Status:** **Rev 2** — Codex round-1 findings addressed (changelog §10); awaiting Yixun approval. No implementation before approval.

**Rev 2 inputs:** Codex plan review round 1 (`yaw_gen_codex_plan_review.md`, verdict REVISE, 8 blocking + 2 nits) · Yixun's three design confirmations (worklog 23:26 EDT: primary = absolute `m_R`; rotation seed = eval seed; V-cells kept as QA gates) · discovery of concurrent exp_15 (yaw-augmentation training) sharing code seams (§9).

---

## 1. Question and estimand

**Question (Yixun, queries 1–2):** Under a physically-consistent *random* yaw shift of the panoramic depth conditioning, how robust is each trained conditioning variant — vanilla vs frame-averaged with group order C4/C8/C16/C32? Expectation to test: robustness improves with group order (vanilla worst).

**Estimand.** For each arm `a ∈ {VANL, C4L, C8, C16, C32}` (the exp_11 8×8 arms, training seed 42, step 40,000) and context size `K ∈ {1, 8}`:

- `m_Z(a, K)` — full-unseen-split per-scene-mean metrics, unrotated (θ = 0), under the arm's own eval protocol (announcement 05: VANL → vanilla conditioning; Cn → `fa_invariant` with the arm's trained orbit `k·360/n`).
- `m_R(a, K)` — the same, but every eval sample independently draws a random yaw `θ_i = d_i · (360/512)°`, `d_i ~ Uniform{0,…,511}` (exact panorama-column roll, no interpolation), applied as the existing physically-consistent rotation (depth roll + per-pixel 3D vectors + all four pose fields rotated together; `context_audio` and GT RIR untouched — a rigid scene rotation leaves the true RIR unchanged).
- **Primary criterion (Yixun 2026-08-10): absolute `m_R`** — which arm performs best when yaw is arbitrary (deployment reading of "outperform"). **Secondary (mechanism): paired degradation** `Δ_m(a,K,s) = m_R(a,K,s) − m_Z(a,K,s)` per eval seed `s`. Both are reported; the headline verdict and the confirmatory inference attach to `m_R` (§4).

Uniform yaw includes each Cn arm's in-group angles as legitimate support mass of the estimand (C4: 4/512 = 0.78% of draws … C32: 32/512 = 6.25% ≈ 396 of 6,337 items per seed). It is never removed arm-by-arm; its expected contribution to each arm's flatness is disclosed in the analysis. *(Review N9.)*

## 2. Why this experiment (context)

- exp_02: vanilla FLAC is **not** yaw-invariant (rel-L2 0.19–0.22 under C4 rotations; T60 gap ~3.4 pp).
- exp_11: trained the orbit arms and closed the θ=0 trend (headline metrics *degrade* with orbit size at matched steps — C8/C16 strictly dominated at this budget). Its yaw-flatness round **R3 produced zero rows**: all 17 screens died on a worktree-provisioning infra failure (`weights/FLAC/VAE.ckpt` leaf symlink missing at pin `75f5934`; fixed by `16b6a20`+`868e93a`, ancestor of HEAD — full forensics in `yaw_gen_worklog.md`). The mechanism the orbits exist for was never measured.
- Honest framing: given exp_11's θ=0 trend, the plausible outcome space includes "higher order is flatter (small Δ) yet still worse in absolute `m_R`". The design reports absolute and paired readouts separately and never blends them.
- Side product: the θ=0 VANL cells populate the still-missing vanilla baseline row in `model_comparison.md` (§7).
- Relation to exp_15 (concurrent, planned): exp_15 trains a *yaw-augmented* vanilla arm and will measure its robustness **under this experiment's protocol** — exp_14 defines and owns that protocol (§9).

## 3. Design

### 3.1 Cells (106 single-GPU jobs, one campaign pin)

| Block | Cells | Protocol | Count |
|---|---|---|---|
| **R** (headline) | 5 arms × K∈{1,8} × eval seeds 42–46 | random per-sample yaw, rotation seed = eval seed | 50 |
| **Z** (reference) | 5 arms × K∈{1,8} × eval seeds 42–46 | θ = 0 | 50 |
| **V** (validity, s42/K8 only, never headline) | `{C4L,C8,C16,C32}@90°` (in-group floor gates) · `VANL@90°` (harness positive control — exp_02 prior guarantees degradation) · `C4L@45°` (**mechanism control only, not a gate** — an fa-trained arm may legitimately be robust off-group; its outcome is reported descriptively) | fixed `--rotate-deg` | 6 |

*(Review B2: VANL@90 added as the positive control; C4L@45 demoted to mechanism control.)*

- All 106 cells run at **one** campaign pin. Z-cells are re-run at our pin even though Cn θ=0 rows exist from exp_11 — those committed rows are demoted to external reproduction checks (§4 G5), not paired references.
- Z and R for the same (arm, K, seed) differ only in rotation ⇒ Δ is seed-paired. R cells across *arms* at the same (K, seed) share the identical per-item rotation assignment (§3.3) ⇒ cross-arm contrasts are rotation-matched.
- V-cells double as the validation-ladder smoke/timing probes (§6).

### 3.2 Protocol constants (identical to exp_11 conf cells, plus two new pins)

Full published unseen split (announcement 01: 6,337 items / 17 rooms; K=8 `acousticroom_unseeneval.json`, K=1 `acousticroom_unseeneval_1.json`), EMA weights, cfg 1.0, 1 diffusion step, bf16 conditioning autocast, per-scene-mean aggregation. **New pins (review B1): eval `batch_size` and `num_workers` are fixed campaign constants for every cell** (values = the exp_11 conf-cell values, read from the kit at adaptation time and recorded in each manifest; the collector rejects any cell whose manifest deviates). Checkpoints and identity gates come from exp_11's committed `arm_launch_registry.json` (read-only; sha256 re-verified per job by the kit's existing gates). Eval-protocol flags (`cond_method`, `frame_avg_angles`, `rotate_mode`, `rotate_seed`/`rotate_deg`) recorded in every submission manifest (announcement 05).

### 3.3 Random-yaw drawing scheme and assignment integrity (pre-registered; review B1)

- **Draws:** one dedicated `torch.Generator(device='cpu')` seeded with rotation seed = eval seed; per batch of size B, the next B integers `d_i ∈ {0,…,511}`; item i's angle is `d_i · 2π/512` exactly (`rotate_scene_metadata` re-quantizes to the same grid — exactness is structural). The dedicated generator must not advance global RNG state (tested).
- **Stream determinism:** `shuffle=False`, `drop_last=False` (verified in code by the reviewer); the draw is a per-item stream, so any batch size partitions the same sequence (tested) — but batch size is pinned anyway (§3.2).
- **Assignment integrity — hashes over what actually happened, not the RNG stream alone.** The dataset can recursively substitute items on load/silence failure, and the metadata module draws context sources stochastically, so an offset-only hash is insufficient. Every cell's sidecar therefore records, per stream position `i`: the target item identity (dataset index + relpath), the ordered context-source identity list as exposed in that sample's metadata (the context fingerprint), `img_w`, and — in R/V cells — the applied offset `d_i`. Two canonical hashes are stored:
  - `input_hash` = sha256 over the ordered `(i, target_relpath, context_ids, img_w)` tuples;
  - `assignment_hash` (R/V only) = sha256 over the ordered `(i, target_relpath, d_i)` tuples.
  Canonical serialization: one JSON array per tuple, `json.dumps(..., sort_keys=True, separators=(",", ":"))`, LF-joined, UTF-8. Expected tuple count: exactly 6,337; the sidecar also stores the count.
- **Substitution guard:** the cell fails (status FAILED, not a silent pass) unless the stream has exactly 6,337 positions and every position's target relpath matches the split manifest order precomputed from the dataset config.
- **Fail-closed equality checks (collector, before any contrast):** (a) across arms within (K, seed): `input_hash` equal for all 5 arms, and `assignment_hash` equal for all R cells; (b) within (arm, K, seed): `Z.input_hash == R.input_hash` (pairing validity). Any inequality → the affected contrast renders BLOCKED, never a number.
- 5 eval seeds ⇒ 5 independent rotation assignments; sampling noise and yaw-draw variation co-vary by design (Yixun-confirmed); the paired Δ removes the shared sampling-noise component at matched seed.

## 4. Hypotheses, statistics, decision rules (pre-registered; review B3)

**Estimation conventions (all confirmatory contrasts):** 5 seed-paired observations; the per-seed observation is the per-scene-mean aggregate for that (arm, K, seed) cell; estimate = mean of the 5 per-seed differences; two-sided 95% paired-t CI with df = 4; α = 0.05. `|Δ|` = absolute value of the per-seed aggregate change `Δ_m(a,K,s)`. **Metric directions (lower = better unless stated):** T60% ↓, C50 ↓, EDT ↓, FD ↓, R@1 ↑, R@5 ↑, R@10 ↑. Co-primary metrics: **T60%** and **R@1**; Holm correction over the two co-primaries *within* each labeled hypothesis. K=8 is confirmatory; K=1 repeats everything descriptively. Adjacent-order contrasts always use the **fixed order `VANL→C4L→C8→C16→C32`** with unadjusted descriptive CIs and no categorical verdicts; the observed ranking is reported descriptively only (no data-dependent confirmatory contrasts).

- **H-P (PRIMARY — absolute robustness; Yixun 2026-08-10):** endpoint tests `m_R(C32) vs m_R(VANL)`, K=8, on T60% and R@1 (Holm-2). Verdict rule: **SUPPORTED** = both co-primaries favor C32 after Holm; **PARTIAL** = exactly one; **NEGATIVE** = neither (or reversed). Adjacent fixed-order contrasts on `m_R` descriptive.
- **H-M (secondary — mechanism/flatness):** endpoint `|Δ|(C32) vs |Δ|(C4L)`, K=8, T60% and R@1 (Holm-2), same verdict rule; `|Δ|(VANL) vs |Δ|(C4L)` reported alongside; adjacent fixed-order contrasts on `|Δ|` descriptive.
- **H-S (sanity — vanilla non-robustness):** `Δ(VANL) ≠ 0`, K=8, T60% and R@1 (Holm-2). Expected to confirm (extends exp_02 to the random-yaw estimand).

**Validity gates — all executable, all must pass before H-readouts are read (review B2):**

- **G1 (in-group floor):** for each Cn arm: `|m(V@90°, s42, K8) − m(Z, s42, K8)| ≤ 0.5 · σ̂_m(arm)` per co-primary metric, where `σ̂_m(arm)` = std over that arm's 5 Z seeds at K=8 (computable because the Z block completes first). Same-seed pairing makes this deliberately conservative; the expectation is diff ≪ tolerance. Failure ⇒ HALT + triage.
- **G2 (positive control):** `m_T60(VANL V@90°, s42, K8) − m_T60(VANL Z, s42, K8) ≥ 5 · σ̂_T60(VANL)` (degradation direction; exp_02's ~3.4 pp prior sits far above this bound). Failure ⇒ HALT (harness is not detecting non-invariance).
- **G3 (golden assignment):** the smoke R-cell's sidecar offset sequence for rotation seed 42 equals the sequence pre-computed in the unit test (proves the draws reach `rotate_scene_metadata`; backed by a pytest integration spy).
- **G4 (assignment integrity):** every §3.3 hash equality holds.
- **G5 (external reproduction — check, not gate):** exp_14 Z rows for Cn vs exp_11's committed conf rows; mean differences reported, discrepancies beyond `3·√(σ_11² + σ_14²)/√5` disclosed in the analysis (cross-pin, does not halt).
- C4L@45° carries **no** gate role (an fa-trained arm may legitimately be robust there); it is reported in the validity section as mechanism context.
- No new eval configurations (announcement 01): the rotation perturbs conditioning of the existing full split; the item set is untouched.

## 5. Implementation plan (per file)

Role split per SOP: Opus 5 max-effort Coder implements; Codex gpt-5.6-sol xhigh reviews per round; TDD (red→green, one small commit per cycle) for every new function; tests in `src/tests/`.

### 5.1 `src/data/yaw_rotation.py` — extend (≈ +40 lines) — **round 1**

- `draw_yaw_offsets(n, img_w, generator) -> LongTensor[n]` — pure; `torch.randint` on the dedicated generator; never touches global RNG.
- `offsets_to_radians(offsets, img_w) -> list[float]` — exact `d · 2π/W`.
- No changes to existing functions (fixed path untouched). These helpers are the shared seam exp_15 will reuse (§9).

### 5.2 `eval_FLAC.py` — extend (≈ +70 lines) — **round 1**

- New flags: `--rotate-mode {fixed,random}` (default `fixed`) and `--rotate-seed INT` (default: the eval seed). Guards, both hard errors: `random` with nonzero `--rotate-deg`; `--rotate-seed` given explicitly while mode is `fixed` (review B4 — never silently ignored).
- Random path: per batch, `draw_yaw_offsets(len(metadata), 512, gen)` → per-sample `rotate_scene_metadata(md, alpha_i, img_w)` (the existing per-sample comprehension with per-item angles); §3.3 tuple stream accumulated and hashed.
- **Fixed-mode behavior is frozen exactly** (review B4): output paths, metrics-record keys/order, serialized JSON bytes, predictions-meta — all byte-identical to current behavior in fixed mode (including default no-flag invocations). Random-mode provenance (`rotate_mode`, `rotate_seed`, `input_hash`, `assignment_hash`, tuple count, `img_w`; `rotate_deg` recorded as `null`) is added **conditionally, only in random mode**, and lives in the metrics JSON (and the kit's `.screenmeta.json` sidecar). Predictions-meta is extended only for callers that actually pass `--store_predictions` (not this campaign — review B5).
- Naming: random-mode suffix/eval-name token **`_rotrand<seed>`** (e.g. `_rotrand42`) — injective across rotation seeds (review B5).

### 5.3 `src/tests/test_yaw_random_eval.py` — new (TDD, written first) — **round 1**

1. Determinism: same seed ⇒ identical offset sequence; different seeds differ.
2. Chunk independence: draws in chunks of 8 vs 64 give the same stream.
3. Global RNG isolation: `torch.random.get_rng_state()` identical before/after draws (review B1).
4. Exactness: every drawn angle round-trips through `rotate_scene_metadata`'s `dj` with zero re-quantization error.
5. Per-sample application: per-item angles == per-item scalar-path application.
6. **Fixed-mode snapshot matrix** (review B4): `{vanilla, fa} × {rotate_deg 0, 45} × {default flags, explicit --rotate-mode fixed}` — serialized metrics-record JSON bytes, output paths, and rotation dispatch identical to pre-change behavior (golden fixtures captured from the current code before any edit).
7. Guards: `random`+`rotate_deg≠0` raises; explicit `--rotate-seed` in fixed mode raises.
8. Naming: `_rotrand42` vs `_rotrand43` produce distinct paths under one eval name; injectivity vs `_rot<tok>` fixed tokens.
9. Canonical hash: serialization golden test (known tuples → known sha256); tuple-count enforcement.
10. Golden assignment: precomputed expected seed-42 offset prefix; integration spy asserting those offsets reach `rotate_scene_metadata` (G3's pytest half).
11. Existing eval/yaw regression subset runs green in the same session (ladder rung 2).

### 5.4 Screen kit — `yaw_gen_screen.sbatch`, `yaw_gen_screen_submit.sh`, `yaw_gen_screen_guardtests.sh` — **round 2**

- **Commit A: verbatim copies** of the exp_11 kit (no edits — review diffs isolate our changes; exp_11's files are never touched).
- **Commit B+ (deltas, each < 200 lines):** namespace `exp11_` → `exp14_`; cell types `{rgen, zref, vctl}` — `rgen`: all 5 arms, seeds 42–46, both K, passes `--rotate-mode random --rotate-seed $SEED`; `zref`: θ=0, all 5 arms, seeds 42–46, both K; `vctl`: exactly the six §3.1 tuples, rejected otherwise; VANL registered for `rgen`/`vctl` (exp_11's r3 exclusion was campaign-specific); `EVAL_ORBIT`/cross cells dropped; batch-size/worker pins exported and recorded (§3.2); eval names `exp14_<ARM>_<CELL>[_rot<TOK>|_rotrand<SEED>]_S<STEP>_s<SEED>_K<K>` (STEP included — fixes the r3 schema's omission). All fail-closed gates kept: pin-worktree leaf-asset provisioning (incl. `VAE.ckpt` — the R3 root cause), registry/manifest sha re-verification, checkpoint identity gate, dual-tee logs, atomic manifests, lease files.
- Guardtests (review B8): assert the exact allowed V tuples, the complete unique 106-cell grid, and rejection of every unregistered (arm, cell, angle, seed, K) combination.

### 5.5 `yaw_gen_submit_grid.sh` — new (≈ 100 lines) — **round 2**

Wave submitter over the 106-cell grid: bounded concurrency (≤ 16 queued+running via `squeue -u`), `DRYRUN=1` printing the full grid (guardtest-compared against the expected set), node `EXCLUDE` passthrough. **Dedup is validate-before-skip (review B6):** a cell is skipped only when its existing metrics JSON + sidecar pass the full §3.3/§5.6 validation (pin sha, checkpoint sha, protocol flags, 6,337 count, hashes); an artifact that exists but fails any check **halts the submitter for triage** — no skip, no overwrite. In-flight cells recognized from kit lease files + `squeue`, reported separately. Every submission appended to `yaw_gen_command.md` at launch time.

### 5.6 `yaw_gen_collect.py` — new (≈ 250 lines) — **round 3** — per-function TDD (review B8)

| Function | Red tests (first) |
|---|---|
| `parse_cell_artifact(path)` | malformed JSON, missing keys, wrong schema version |
| `validate_cell_provenance(rec, expected)` | wrong pin / ckpt sha / protocol flags / batch pins / tuple count each rejected with a named reason |
| `expected_grid()` | exactly 106 unique cells; V tuples exact; no unregistered combos |
| `match_assignments(records)` | cross-arm `input_hash`/`assignment_hash` and Z↔R `input_hash` violations detected and named |
| `pair_seeds(z_cells, r_cells)` | missing seed, duplicate seed, orphan cells rejected |
| `paired_t_ci(diffs, alpha)` | known-value fixtures (cross-checked against scipy) |
| `holm_adjust(pvals)` | known fixtures incl. ties |
| `metric_direction(metric)` | complete table per §4 |
| `aggregate_cell(seed_records)` | 5/5 enforcement; PENDING rendering for partial cells (never numbers) |
| `evaluate_gates(cells)` | G1–G4 pass/fail on synthetic fixtures; BLOCKED rendering on failure |
| `suppress_validity_cells(rows)` | V-cells never appear in headline tables |
| `render_tables(results)` | golden markdown fixture |

Output: `_results.md` tables (absolute R, paired Δ, endpoint contrasts with Holm, descriptive adjacent CIs, gate report) + a JSON bundle for the HTML.

### 5.7 `model_comparison.md` integration — **round 3** (review B7)

`gen_model_comparison.py` gains an **explicit exp_14 row-spec contract** (glob + validation for `exp14_VANL_zref_*` metric JSONs), with tests. The never-populated exp_11 Q9 VANL row spec is **replaced** by the exp_14-sourced spec, clearly relabeled (`fa-recipe vanilla VANL @40k (exp_14 Z, one-pin)`); exp_11's populated rows and its validator are untouched. The within-pin fa-vs-vanilla delta that Q9 wanted is delivered inside exp_14's own results (all five arms share one pin here — this supersedes the never-run Q9 estimand, stated as such in `_results.md`). Regenerate + commit + push fires **immediately when the VANL Z 2-K × 5-seed transaction completes** (a §6 sequencing step, not a post-results afterthought).

### 5.8 Later phase (post-results, per SOP)

`yaw_gen_results.md`, `yaw_gen_analysis.md`, `yaw_gen_01_results.html` + assets (dataviz guidance loaded before chart code), `commits_yaw_gen.md`.

## 6. Validation ladder (cheapest-first, each rung logged in worklog)

1. Static: `py_compile` / `bash -n` / `git diff --check` on every changed file.
2. Pytest: §5.3 + §5.6 suites and the existing eval/yaw regression subset green.
3. Kit guardtests + `DRYRUN=1` submitter: printed grid == `expected_grid()` exactly.
4. **Smoke = V-cells** (6 jobs, ~2.5 GPU-h): `C4L@90` runs first alone — exercises worktree provisioning (VAE.ckpt leaf-asset regression from R3), the fa path, fixed rotation, metrics landing. Then the other five V-cells. Read as G1/G2 gates *and* per-cell timing.
5. One `rgen` probe (C32, K=8, s42 — most expensive class): times the random path, and its sidecar seeds G3.
6. Z block waves, then R block waves, ≤ 16 concurrent; dedup skips validated probe/V cells. VANL Z transaction completion triggers §5.7 table regen.
7. Gates G1–G4 evaluated; only then H-readouts.

## 7. Cost and schedule

Per-cell wall-times from exp_11 Slurm accounting (1× L40, 10 CPU, 32 GB — taken from `sacct`, not re-measured): K=1 ≈ 13 min, K=8 ≈ 23 min (C16), C32 up to ~36 min. Estimate: 106 cells ≈ **30–45 GPU-h**. Wall-clock ≈ **4–6 h of queue-excluded service time** at ≤16 concurrent slots (review N10: actual elapsed time depends on queue depth; L40 availability not guaranteed). No training; no multi-GPU jobs. Coder + review rounds realistically dominate (~2–4 h before first launch).

## 8. Risks / standing-rule compliance

- **Concurrent writers:** `git pull --rebase` before every commit; after submitting pin-bound jobs, **no tracked-file changes until every job passes its start gate** (exp_11 standing rule; three prior incidents). exp_11's folder and files are read-only to this experiment.
- **Protocol-flag trap (announcement 05):** every manifest records `cond_method`, `frame_avg_angles`, `rotate_mode`, `rotate_seed`/`rotate_deg`; the collector re-validates per cell class before aggregation.
- **Legacy behavior frozen:** fixed mode byte-identical (snapshot-tested, §5.3.6); no other experiment's tooling can be perturbed.
- **Storage:** no `--store_predictions` anywhere; metrics + sidecars only (~KB–MB each; headroom 1.6 T per `df`).
- **Codex sandbox:** every review prompt forbids installing/modifying environments (`-s read-only` is not sufficient protection).

## 9. Coordination with exp_15 (concurrent session, yaw-augmentation training)

exp_15 (scaffolded 2026-08-10 by the concurrent session; training-side random-yaw augmentation for a vanilla arm) **consumes** exp_14's protocol: its robustness cells use this experiment's random-yaw eval and its augmentation policy mirrors §3.3's drawing conventions. Seam ownership, recorded here so both sessions see it:

- **exp_14 owns** the eval-side extension: `eval_FLAC.py` random mode and the shared pure helpers `draw_yaw_offsets` / `offsets_to_radians` in `src/data/yaw_rotation.py`. These land first (round 1).
- **exp_15 owns** the training-side augmentation hook and must build on exp_14's committed helpers (pull-rebase; no parallel edits to `yaw_rotation.py` or `eval_FLAC.py` while exp_14's rounds are open).
- Neither session edits the other's experiment folder.

## 10. Rev 2 changelog (review finding → change)

| Finding | Disposition |
|---|---|
| B1 assignment integrity | §3.2 batch/worker pins; §3.3 per-position tuples, `input_hash`/`assignment_hash`, substitution guard, collector equality checks; §5.3.3 global-RNG test |
| B2 gates not executable | §4 G1–G5 with pre-registered formulas; VANL@90 positive control added (grid 105→106); C4L@45 demoted to mechanism control; G3 golden-assignment gate |
| B3 stats incomplete / data-dependent contrasts | §4 conventions block (df=4 paired-t, directions, \|Δ\| definition, verdict rules); fixed-order adjacent contrasts, observed ranking descriptive-only |
| B4 byte-compat contradiction | §5.2 conditional provenance (random mode only); fixed-mode snapshot matrix §5.3.6; `--rotate-seed` in fixed mode errors |
| B5 naming/sidecar ambiguity | `_rotrand<seed>` token; hashes in metrics JSON + `.screenmeta.json`; predictions-meta untouched; canonical serialization + count defined (§3.3) |
| B6 fail-open dedup | §5.5 validate-before-skip; halt-for-triage on invalid artifacts; leases for in-flight |
| B7 VANL table contract | §5.7 explicit exp_14 row-spec contract + tests; Q9 spec replaced + relabeled; immediate regen trigger; within-pin delta supersedes Q9 in exp_14 results |
| B8 per-function TDD | §5.6 function/test table; §5.4 guardtest grid assertions |
| N9 in-group framing | §1 support-mass framing + disclosure |
| N10 wall-clock realism | §7 queue-excluded service-time label |
