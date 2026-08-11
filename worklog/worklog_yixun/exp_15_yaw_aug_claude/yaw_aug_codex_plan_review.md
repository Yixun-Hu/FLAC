# exp_15 yaw_aug — Codex PLAN review

**Reviewer:** OpenAI Codex `gpt-5.6-sol` at Extra High reasoning (codex-cli 0.146.0, `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh`, read-only sandbox; the model self-reported generically as "GPT-5, API workspace agent") · **Date:** 2026-08-10 · **Object under review:** `plan_yaw_aug.md` Rev 1 · **Verdict: REVISE** — addressed in Rev 2 (§11 changelog maps every finding to its disposition).

# exp_15 `yaw_aug` plan review

**Reviewer:** OpenAI Codex (GPT-5, API workspace agent; exact serving subversion/reasoning tier not exposed; read-only inspection) · **Date:** 2026-08-10

The experiment question is sound, and the four Yixun-confirmed decisions are respected. The full-split Table-1 design also complies with announcement 01. However, the plan is not implementation-ready.

## Findings

1. **BLOCKING — exp_15 does not actually adopt exp_14’s shared evaluation contract.**

   Exp_15 specifies only an offset-stream hash (`plan_yaw_aug.md:61,102,126`). Exp_14 requires `input_hash`, item-bound `assignment_hash`, exactly 6,337 tuples, substitution detection, cross-arm input equality, and Z↔R input equality (`plan_yaw_gen.md:49-58,93`). An offset-only hash cannot detect the recursive sample substitution present in `SampleDataset.__getitem__` (`src/data/dataset.py:235-310`) or changed context selection.

   There is also an ownership conflict: exp_15 says “first lander owns” (`plan_yaw_aug.md:147`), while exp_14 explicitly owns `eval_FLAC.py` and the shared yaw helpers and requires exp_15 to build on its reviewed commits (`plan_yaw_gen.md:169-175`).

   Finally, exp_14’s proposed context-identity hash assumes context IDs are exposed, but AR metadata currently exposes only context tensors—not selected source paths/IDs (`src/configs/dataset_configs/custom_metadata/AR_md.py:39-45`). Target `idx`/`relpath` are available, but context identity is not (`src/data/dataset.py:267-289`).

   **Fix:** Make exp_14 the sole owner and a hard dependency: exp_15 must pin and reuse exp_14’s closed, reviewed round-1 commit. Before that round lands, amend the shared contract to either expose ordered context IDs with TDD and fixed-mode regression coverage, or define a canonical fingerprint using data that actually exists. Exp_15 must adopt the exact hashes, tuple count, substitution guard, batch/worker pins, `_rotrand<seed>` naming, and Z↔R equality checks. No exp_15 edit to `yaw_rotation.py` or `eval_FLAC.py`.

2. **BLOCKING — the historical VANL checkpoint is not checkpoint-SHA-pinned as claimed.**

   The plan says the 40k checkpoint is SHA-pinned in exp_11’s registry (`plan_yaw_aug.md:14,122`). The registry’s VANL entry pins the launch manifest, commit, config, VAE, rung, and seed, but contains no `final_ckpt_path` or `final_ckpt_sha256` (`arm_launch_registry.json:107-123`). Disk presence alone does not provide immutable control identity, especially because the Slurm job ended `FAILED` after saving.

   **Fix:** Without editing exp_11, create an exp_15-owned immutable control-admission record that binds:

   - Exact canonical checkpoint path and SHA-256.
   - Embedded step = 40,000 and embedded model config/config hash.
   - Presence and readability of EMA, optimizer, and scheduler state as appropriate.
   - Source exp_11 launch-manifest SHA, commit `81ddac3…`, training seed, rung, and VAE hash.

   The eval kit must re-hash and preflight this record before every VANL cell.

3. **BLOCKING — a desired scientific outcome is incorrectly used as a validity gate.**

   Requiring YAWAUG@90° to sit “near its own θ=0 floor” before reading H1–H3 (`plan_yaw_aug.md:77`) assumes the augmentation succeeded. Uniform training exposure does not guarantee a learned invariant model. A valid negative H2 must remain reportable. Likewise, “if H2 fails, implementation is suspect before the science” (`plan_yaw_aug.md:74`) establishes a confirmation-biased triage rule.

   The cross-pin VANL reproduction check is also described as a halt gate, although an honestly measured cross-pin discrepancy does not invalidate the within-pin exp_15 comparison when provenance is otherwise correct.

   **Fix:** Make YAWAUG@90 descriptive/mechanistic, never gating. Integrity gates should be limited to executable harness checks: VANL positive control with a numeric threshold, golden random assignment, full input/assignment hash integrity, protocol/checkpoint validation, and cell completeness. Treat historical-row reproduction as a non-halting external check with a predeclared formula, following exp_14 G5 (`plan_yaw_gen.md:71-75`).

4. **MAJOR — the statistical rules are incomplete and overstate what five eval seeds estimate.**

   `σ_c` and the exact SUPERIOR/EQUIV/INFERIOR rules are undefined (`plan_yaw_aug.md:71-75`). “Within 1σ” is not a statistical equivalence test. H2 does not say whether `|Δ|` is taken per seed or after averaging, H3 has no categorical decision rule, and three separately Holm-corrected hypotheses do not control multiplicity across the complete claim family.

   More importantly, both arms have one training run. Five evaluation seeds estimate diffusion/rotation-assignment variability, not training-run variability. Matching training seed and step does not pair away checkpoint-band variability (`plan_yaw_aug.md:151`).

   **Fix:** State:

   - Per-seed observation, metric direction, paired difference, df=4 paired-t CI, alpha, and exact Holm family.
   - `|Δ|` as the absolute per-seed aggregate change before contrasting arms.
   - Exact SUPPORTED/PARTIAL/NEGATIVE or INCONCLUSIVE rules.
   - If “EQUIV” is retained, a preregistered practical margin and a valid equivalence procedure; otherwise use “not statistically resolved.”
   - H1 as confirmatory and H2/H3 as secondary, or define one multiplicity family covering all confirmatory tests.
   - All inference is conditional on these two seed-42 training trajectories; no training-run variance is estimated.

5. **MAJOR — resume behavior violates the fresh-draw policy and the determinism claim is unsupported.**

   The plan admits the generator is not checkpointed and will re-seed on restart, but calls the repeated stream “fresh” (`plan_yaw_aug.md:37-38`). A resumed leg will replay the per-rank offset-stream prefix. The claim that Lightning restores no relevant RNG/dataloader position is not established by repo code; the repo simply passes `ckpt_path` to `trainer.fit` (`train.py:230`).

   **Fix:** Make restarts exactly reproducible using either:

   - A counter-based stream keyed by augmentation seed, global step, global rank, and within-batch index; or
   - Properly gathered/restored per-rank generator states in the checkpoint.

   Pin world size and micro-batch across resume. Add tests for uninterrupted `N+M` draws versus save/restore after `N`, rank mapping, and isolation of Python/NumPy/torch global RNG states during generator construction, drawing, and augmentation.

6. **MAJOR — the historical-control no-op audit is too weak.**

   The current read-only diff from control commit `81ddac3…` to inspected HEAD `89f24cd…` is favorable: no production training files changed, only tests. But the eventual exp_15 pin will change the factory and training wrapper. Metadata object identity alone (`plan_yaw_aug.md:108-113`) does not establish whole-step equivalence, and YAWAUG-vs-VANL W&B loss curves cannot diagnose a no-op regression because the treatment intentionally changes conditioning and loss (`plan_yaw_aug.md:48,150`).

   **Fix:** Add:

   - A fail-closed launch-pin diff allowlist from `81ddac3…`.
   - A pre-edit golden deterministic disabled-path test covering conditioning output, loss, global RNG states, and relevant wrapper state.
   - Conditional omission of new factory kwargs when `yaw_aug` is absent.
   - Environment/hash comparison against the exp_11 manifest.

   Describe VANL as a “historical recipe-matched control,” not a strictly contemporaneous single-delta control.

7. **MAJOR — hook placement is reasonable, but runtime/schema guards are missing.**

   Applying augmentation immediately before `_compute_conditioning` is the right conceptual seam (`src/training/diffusion.py:223-246`), and `rotate_scene_metadata` correctly shallow-copies metadata while rotating depth and the four pose fields (`src/data/yaw_rotation.py:224-242`). However, it trusts configured `img_w` when quantizing the angle and rolls the actual tensor without checking that its width matches.

   The plan also estimates work on a `3×64×512` tensor (`plan_yaw_aug.md:137`), while AR metadata constructs `3×256×512` depth (`AR_md.py:47-52`). Calling the operation “CPU-cheap” is also an unverified device-placement assumption at `training_step`.

   **Fix:** Add fail-closed validation for nonempty metadata, depth presence/shape, width exactly 512, positive integer `img_w`, true boolean `enabled`, valid integer seed, and expected pose trailing dimensions. Correct the cost model and benchmark the actual `3×256×512` path. Add fixed-offset integration cases `{0,1,128,511}` verifying all four poses, no input mutation, unchanged `reals/context_audio/padding_mask/scene`, dtype/device preservation, and width-mismatch rejection.

8. **MAJOR — mandatory TDD and validation-ladder coverage is incomplete.**

   The collector is described only as “pure functions tested” (`plan_yaw_aug.md:124-126`), not with the per-function test inventory required by announcement 02 (`announcement/02_test_driven_development.md:11-15`). The validation ladder jumps from tests/guardtests directly to an 8-GPU smoke, omitting the SOP’s tiny synthetic forward and small real-data readback (`experiment_SOP.md:69-79`).

   **Fix:** Enumerate collector functions and red tests for parsing, provenance, exact 42-cell grid, input/assignment matching, seed pairing, paired-t CI, Holm, metric directions, completeness, gates, and rendering. Add:

   - A tiny real-wrapper synthetic training step.
   - A few-record AR readback checking actual metadata shapes/dtypes and rotation invariants.
   - Exact-grid and reject-all-unregistered-cell guardtests.
   - A disabled-path regression subset and complete existing pytest subset.

9. **MAJOR — announcement 05 requires explicit CLI flags, not merely manifest fields.**

   The plan records protocol fields but does not prescribe every required flag in every invocation (`plan_yaw_aug.md:63,120-122`). Current defaults are specifically unsafe: `--frame-avg-angles` defaults to C4 and `--cond-autocast` defaults to `default`, not bf16 (`eval_FLAC.py:483-486`).

   **Fix:** Pin and guard the literal argv:

   - T: `--cond-method vanilla --frame-avg-angles 0,90,180,270 --cond-autocast bf16 --rotate-mode fixed --rotate-deg 0`.
   - R: the same conditioning flags plus `--rotate-mode random --rotate-seed <eval-seed> --rotate-deg 0`.
   - V: fixed mode, `--rotate-deg 90`, and no random seed.

   Record and revalidate all flags in plan, params, command log, screen manifest, metrics JSON, and collector.

10. **MAJOR — announcement 04 timing and VANL row ownership are unresolved.**

    The plan defers table integration to “post-results” (`plan_yaw_aug.md:128-130`), while announcement 04 requires immediate regenerate/commit/push on every completed five-seed model-results block (`announcement/04_model_comparison_table.md:3-5`). The generator already contains pending exp_11 VANL row specs (`gen_model_comparison.py:82-94`), while exp_14 preregisters replacing those with exp_14 Z cells (`plan_yaw_gen.md:139-141`). “Add VANL if still absent” risks conflicting row ownership.

    **Fix:** Pre-register the transaction trigger: when YAWAUG T has 5/5 seeds at both K, land the tested row spec, regenerate, commit, and push immediately. Exp_14 should own the existing VANL row unless both plans explicitly agree on a different authoritative source. Exp_15’s fresh VANL cells can remain a results-local reproduction/control block without creating a duplicate model row.

11. **MAJOR — the copied training kit currently encodes a different budget.**

    The source exp_11 launcher now pins 100,000 steps and explicitly rejects any other production maximum (`fa_orbit_train.sbatch:71-75,187-190`), although its argv structure remains at `:293-305`. The exp_15 delta list does not explicitly enumerate removal of these extension-era assumptions.

    **Fix:** List and guard every required change: `PINNED_MAXSTEPS=40000`, the production max-step assertion, 40k INITIAL/RESTART semantics, time pins, legal-arm set, output paths, registry schema, and the exact argv diff. Add guardtests proving production cannot launch at 100k and a restart cannot exceed the preregistered 40k endpoint.

12. **MINOR — bookkeeping and terminology need tightening.**

    The master tracker still says nothing is in flight and has no exp_14/exp_15 rows (`master_experiment_tracker.md:27`). The plan also uses “byte-identical” for wrapper construction where behavioral/golden equivalence is the supportable claim.

    **Fix:** Add the planned/in-review exp_15 lifecycle entry during the plan amendment and coordinate the exp_14 tracker entry. Reserve “byte-identical” for serialized artifacts actually covered by byte snapshots; use “behaviorally identical under the disabled-path regression” elsewhere.

## Hidden assumptions / failure modes to make explicit

- Random-yaw pairing is valid only if actual item and context assignments match—not merely offset streams.
- Dataset recursion can silently replace failed/silent examples; tuple count alone does not detect identity substitution.
- Context-source selection uses NumPy RNG and currently lacks source identifiers in returned metadata.
- Five eval seeds do not provide uncertainty over training initialization, data order, hardware timing, or checkpoint-band position.
- A restart or topology change can alter both data visitation and yaw assignments unless the stream is resume-exact.
- The startup log line proves a branch was entered, not that every sample was correctly transformed; unit/integration evidence remains necessary.
- The historical control remains noncontemporaneous even if the disabled path is proven equivalent; this limitation must constrain causal language.
- The 24-hour estimate depends on measured throughput of the actual 256×512 batched metadata path.

**VERDICT: REVISE**