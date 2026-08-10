# Re-pin candidate review — Q10 + chain-gate + K1 rounds (45b6154 era)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-10

# exp_11 Q10/K1 re-pin review

**VERDICT: NOT APPROVED** for `45b61540db5eaf6952edfcc14f6a33ad1eebd495` as the new campaign pin. Keep the campaign pin at `0c6e9ffb616cbd788b420e67d62638ad40a7b13c`.

`f893037` and `3b013ce` are not ancestors of current HEAD, but their stable patch IDs exactly match the branch copies `e661f19` and `75a1cd2`, respectively; those copies are present in the candidate.

## Blocking findings

1. **The queued Q10 restart jobs have three independent hard-abort paths.**

   - Jobs 3662828–3662830 were submitted with `EXPECT_SHA=c85bc61`, while the mutable training checkout is now at `45b6154`. Training is not worktree-pinned, and the launcher aborts when repository HEAD differs from `EXPECT_SHA` at [fa_orbit_train.sbatch:183](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:183).
   - The submitter correctly allocates the RESTART limits—34/51/89 hours—but the job itself always selects the INITIAL per-arm limit at [fa_orbit_train.sbatch:159](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:159) and then rejects the allocated RESTART limit at [fa_orbit_train.sbatch:388](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:388).
   - The restart preflight compares the INITIAL manifest’s `max_steps=40000` and original commit `2b78f99` against the extension target `100000` and restart commit at [fa_orbit_ckpt_preflight.py:81](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_ckpt_preflight.py:81). Reproduction returns both mismatches.

   Required fix: select and enforce the RESTART time pin inside the job; give the 40k→100k extension its own reviewed preflight contract that preserves the original 40k launch identity without requiring its budget/commit to equal the extension’s; then cancel and resubmit 3662828–3662830 at the final reviewed SHA. The numeric limits themselves are below the live partition cap (`MaxTime=7-00:00:00`; C32 is 160 hours).

2. **The >40k lineage gate is existential, not checkpoint-specific; a wrong-lineage checkpoint can slip.**

   Once any registry leg has `mode=RESTART` and the right 40k resume hash, every later checkpoint for that arm passes the chain branch at [fa_orbit_screen.sbatch:489](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:489). The gate does not bind the evaluated checkpoint’s SHA or step to that leg, re-hash the registered restart manifest, or check the leg’s job, commit, arm, config, save directory, expected step, or target budget. After one legitimate row exists, a same-config checkpoint from a wrong restart copied into the canonical directory passes.

   Required fix: bind each admissible `step → checkpoint_sha256` to a fully validated restart leg, or provide an equivalent immutable per-checkpoint producer manifest. The screen must verify that exact binding, not merely find one acceptable restart somewhere in the arm’s registry.

3. **The restart recorder is fail-open when the resume file cannot be resolved.**

   [fa_orbit_record_restart.py:53](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_record_restart.py:53) re-hashes only when `os.path.isfile(resume_path)` succeeds; otherwise it trusts the mutable manifest’s claimed hash. It also does not verify manifest arm, canonical location, required job/UUID, expected step 40000, max steps 100000, config/rung/save directory, commit, or pinned time limit. The screen subsequently ignores almost every recorded field.

   Required fix: require the canonical resume file to exist and always re-hash it; validate all restart-manifest identity fields against the INITIAL registry and Q10 pins; publish atomically; reject duplicates; and have the screen re-hash and consume the registered manifest.

4. **The claimed non-gate trajectory separation is not mechanically true.**

   - `gate_admissible(8, contract="traj")` currently returns `True` because [exp11_validate_rows.py:205](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:205) falls back from missing `gate_K` to ordinary `K`.
   - The regression assertion is deliberately vacuous through `or True` at [test_exp11_validate_rows.py:911](/n/fs/gatrdp/codespace/FLAC/src/tests/test_exp11_validate_rows.py:911).
   - A K=1 `CELL=screen` still emits `validated=futility` at [fa_orbit_screen.sbatch:653](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:653), even though K=1 is declared figure-only and non-gate.

   Required fix: make gate admissibility explicit and false for every non-futility contract, remove the vacuous assertion, and emit an explicit non-gate/figure-only status for K1 screens and all `traj` rows.

5. **The figure generator does not enforce the five-seed/provenance rule it claims.**

   [gen_trajectory_figures.py:69](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/gen_trajectory_figures.py:69) draws a band whenever five matching files exist. It does not prove unique seeds 42–46, run `contract="traj"`, inspect sidecars, or require one checkpoint/config/source SHA. Five duplicates or mixed-pin/mixed-checkpoint files can therefore become a band.

   The validator is also incomplete for Q10: it enforces only `step > 40000`, not the 2500 grid or the 100000 ceiling, at [exp11_validate_rows.py:644](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:644). Additionally, the HTML/SVG y-range excludes band values at [gen_trajectory_figures.py:99](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/gen_trajectory_figures.py:99), so future extension bands can render outside the plotted scale.

   Required fix: harvest through `validate_cell(..., contract="traj")`, require the exact five unique seeds and common provenance, enforce the upper bound/grid in the validator, and include band extrema in SVG scaling.

6. **The current immutable pin cannot support the planned trajectory campaign.**

   The pinned registry has `restarts: {}` and C32 has no audited 40k anchor at [arm_launch_registry.json:79](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json:79). Because screens read the registry from the detached pinned worktree, recording restart rows later in the main checkout will not make them visible at `45b6154`. Every >40k trajectory evaluation would therefore fail closed at this pin, and C32 can never chain from it.

   Required fix: wait for and audit C32@40k, add its anchor, record the validated restart legs, and include those records in the final single-pin candidate before q9/traj/VANL submission.

## Verified sound portions

- Table isolation is sound: published exp_11 rows use exact `conf`/`q9` globs and strict cell contracts; no active `traj` path enters the table.
- The q9 four-cell, one-`source_sha` transaction remains intact.
- The trajectory block submitter routes real submissions through the locked campaign-pin/lease path.
- Driver-side trajectory seeds, K values, `>40000`, `≤100000`, and 2500-step cadence checks are correct.
- Focused read-only tests passed: `157 passed`; `git diff --check` was clean. Those tests do not cover the blockers above.
