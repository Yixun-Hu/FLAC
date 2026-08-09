# Code re-review — exp_11 round-4 fixes (967f5f9..25ac371)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** danger-full-access (bwrap unavailable); read-only · **Date:** 2026-08-07

# Re-review — exp_11 round-4 fix series

**Reviewed HEAD:** `25ac371518022d7e118b6e3917a197542a7a4913`, matching `origin/check-equivariance-necessity`  
**Verdict:** **REJECT**  
**Submission decision:** **NO-GO** for the combined backfill + arm-screen series from this HEAD.

The EMA and backfill fixes are substantially better, but the validator/table boundary remains fail-open in several material ways.

## Prior findings

| Finding | Status | Evidence and judgment |
|---|---|---|
| **B1 — EMA claimed, not proven** | **CLOSED** | `resolve_weights_source()` uses the same `use_ema && diffusion_ema.ema_model.*` predicate as the actual swap branch ([eval_FLAC.py:65](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:65), [eval_FLAC.py:277](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:277)). The branch overwrites `model.*` with EMA tensors before checked loading ([eval_FLAC.py:281](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:281), [eval_FLAC.py:289](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:289)); it is not a parallel computation that can presently diverge from the evaluated weights. Screens also refuse EMA-less checkpoints ([fa_orbit_screen.sbatch:198](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:198)), and the validator requires the evaluator’s recorded source to be `ema` ([exp11_validate_rows.py:257](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:257)). |
| **B2 — full split/runtime protocol unproven** | **PARTIALLY CLOSED** | The evaluator now records runtime arguments and actual count ([eval_FLAC.py:424](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:424)), and `n_samples == 6337` is enforced ([exp11_validate_rows.py:260](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:260)). But record↔sidecar checks are skipped whenever the evaluator field is absent/`null` ([exp11_validate_rows.py:273](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:273)); missing `seed`, `cfg_scale`, `steps`, `eval_name`, or `dataset_config` can therefore pass. `batch_size` and `device` are not validated, and model-config/checkpoint/dataset hashes are not in the evaluator record at all ([eval_FLAC.py:119](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:119)). |
| **B3 — malformed/mislabeled rows pass** | **PARTIALLY CLOSED** | Duplicate keys and nonstandard constants are rejected ([exp11_validate_rows.py:121](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:121)); required metrics are finite numeric values ([exp11_validate_rows.py:234](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:234)); rotation and exact generated basename are checked ([exp11_validate_rows.py:254](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:254), [exp11_validate_rows.py:310](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:310)). Residuals: extra metric keys are accepted despite the requested exact-six contract; booleans can satisfy `1`/`1.0`; hashes/commits are string-coerced rather than type-checked and SHA-256 fields may be only 40 characters ([exp11_validate_rows.py:293](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:293)); recomputation remains optional. |
| **B4 — caller-controlled cell/seed policy** | **PARTIALLY CLOSED** | Registered purpose contracts correctly fix cell types, seeds, and K ([exp11_validate_rows.py:76](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:76)), and cell identities are compared ([exp11_validate_rows.py:388](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:388)). The requested outer “both K=1 and K=8” table gate is absent: the generator validates each row independently ([gen_model_comparison.py:163](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:163)). The R3 contract also cannot validate the registered five-angle block: all five files use seed 42, while its exactly-once seed logic treats multiple rows as duplicates ([exp11_validate_rows.py:78](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:78), [exp11_validate_rows.py:372](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:372)). |
| **B5 — validator advisory** | **PARTIALLY CLOSED** | A futility screen now runs validation with recomputed hashes before `SCREENRESULT` ([fa_orbit_screen.sbatch:279](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:279)). The generator blocks detected invalid exp_11 rows before aggregation ([gen_model_comparison.py:102](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:102)). However, generator validation omits `verify_hashes=True` ([gen_model_comparison.py:92](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:92)). Confirmatory runs are not validated by the driver, yet still emit `validated=table` in `SCREENRESULT` ([fa_orbit_screen.sbatch:284](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:284), [fa_orbit_screen.sbatch:293](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:293)). |
| **B6 — checkpoint lineage** | **PARTIALLY CLOSED** | The backfill allowlist is sound: it records only 20k/30k with canonical paths and SHA-256 values ([c4_backfill_manifest.json:17](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/c4_backfill_manifest.json:17)), and the driver enforces resolved path plus recomputed hash ([fa_orbit_screen.sbatch:166](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:166)). I independently recomputed both checkpoint hashes, sizes, mtimes, and the config hash; all match. But exp_11 arm checkpoints still are not bound to their existing launch manifests, training seed, or run identity—the active-arm path uses discovery plus embedded-config equality only ([fa_orbit_screen.sbatch:160](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:160), [fa_orbit_screen.sbatch:193](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:193)). The validator also retains substring-based directory checking ([exp11_validate_rows.py:288](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:288)). |
| **B7 — loop/batched disclosure** | **PARTIALLY CLOSED** | Generator source labels historical FA rows `legacy-loop` and contains the requested non-interchangeability header ([gen_model_comparison.py:43](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:43), [gen_model_comparison.py:152](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:152)). The published artifact remains stale and still says only `fa eval` ([model_comparison.md:24](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/model_comparison.md:24)). Thus the future generator is corrected, but the current results table is not. |
| **N1 — drift scope** | **CLOSED** | The experiment directory is recursive again, with narrow exclusions for append-only training/screen logs ([fa_orbit_screen.sbatch:140](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:140)). |
| **N2 — unregistered 10k backfill** | **CLOSED** | Backfill is restricted to 20k/30k ([fa_orbit_screen.sbatch:103](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:103)). |

## Flagged deviation: no pre-eval intent file

**Acceptable in principle, not acceptable as implemented.**

A complete evaluator-authored runtime record, strictly and unconditionally cross-checked against the driver sidecar, can replace a separate intent file for proving the completed evaluation. A second driver-authored file would not by itself protect against a deliberately malicious driver.

The present implementation does not meet that condition:

- Cross-checks explicitly skip missing evaluator fields.
- The evaluator record contains no model-config, checkpoint, or dataset hashes.
- Evaluator `source_sha` is not required to equal sidecar `commit`.
- `source_sha()` is called while constructing the post-evaluation record, so it records HEAD after the run rather than pinning the code identity before evaluation ([eval_FLAC.py:407](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:407), [eval_FLAC.py:424](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:424)).
- `batch_size` and `device` are recorded but unvalidated.

Consequently, the claimed record↔sidecar contradiction is not currently guaranteed.

## Fresh-eyes judgments

### `resolve_weights_source`

**PASS.** It precisely mirrors the actual EMA selection branch and is computed from the same normalized state dictionary immediately before that branch. The subsequent checked `load_state_dict` means an “ema” record corresponds to EMA tensors actually replacing the online model tensors.

### CONTRACTS policy

**INCOMPLETE.** Futility and table seed policies are correct, but there is no two-K outer table transaction, and the R3 contract models one seed occurrence rather than the registered five rotations. `table_admissible` is documentary; it is not itself consulted.

### Audited manifest

**PASS for the historical backfill exact-file allowlist.** Its live config/checkpoint hashes and file metadata match. The seed/lineage prose remains an audited assertion rather than something derivable from the checkpoint, which is reasonable for a committed historical manifest. It does not cure the missing launch-manifest binding for current exp_11 arms.

### Generator BLOCKED rendering

Once a row is recognized as exp_11 and validation returns failure, **aggregate metric numbers cannot render**: `render_row()` returns the BLOCKED line before `agg_files()`.

That guarantee is conditional:

- Recognition relies on `"exp11_"` appearing in the path rather than explicit row metadata.
- The invoked validator does not recompute hashes.
- Validator schema gaps can therefore admit an unproven row before rendering.

### Rotation suffix

**PASS for this repository.** Integer rotations preserve historical names, while fractional values receive distinct `p` forms ([eval_FLAC.py:48](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:48)). No historical fractional `_rot5`/`_rot11`/`_rot22` artifact exists in this tree, so no existing filename changes meaning. Any off-host artifact formerly produced from a fractional angle remains inherently ambiguous, but no current artifact is renamed or reinterpreted here.

### Unregenerated `model_comparison.md`

Not regenerating the numerical table while the exp_10 endpoint JSONs are absent is the correct operational decision: both endpoint globs currently match zero files, so regeneration would destroy published values by replacing them with pending cells.

It is nevertheless not fully safe yet:

- The current table retains the old ambiguous labels, so B7 remains partial.
- The new test imports the generator at collection time ([test_gen_model_comparison_gate.py:26](/n/fs/gatrdp/codespace/FLAC/src/tests/test_gen_model_comparison_gate.py:26)), while the generator writes the table at module scope without an `if __name__ == "__main__"` guard ([gen_model_comparison.py:143](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:143)). Merely running pytest can therefore regenerate and mutate the table despite the deliberate freeze.

## Required before approval

1. Make evaluator runtime fields mandatory and unconditional; validate exact types, exact metric-key set, batch size, device, and `source_sha == sidecar.commit`.
2. Bind active arm checkpoints to their original launch manifests and canonical run directories.
3. Recompute hashes in the generator’s table validation.
4. Add the two-K outer table gate and model R3 as the registered five-angle set.
5. Do not emit `validated=table` for an unvalidated confirmatory row.
6. Put generator execution behind `main()` so importing/testing cannot rewrite the table.
7. Preserve the current table’s numbers while migrating its protocol labels, or defer that migration explicitly until the missing exp_10 evidence is recovered.

**C4 backfill path in isolation is exact-file-bound and technically sound. The arm-screen lineage and shared proof/table surfaces are not. Therefore the requested combined submission remains NO-GO.**
