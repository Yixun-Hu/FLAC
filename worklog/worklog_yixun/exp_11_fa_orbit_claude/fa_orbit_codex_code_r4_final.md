# Final GO-check review — exp_11 nine-item series (339a125..d273b15)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · **Sandbox:** danger-full-access (bwrap unavailable); read-only · **Date:** 2026-08-07

# Independent code review — exp_11 arm screens and table pipeline

**Reviewed commits:** `339a125`, `abb90a5`, `c295c09`, `ced7f53`, `d273b15`  
**Reviewed HEAD:** `d273b15392edba7ef85930eec4b0ba89916127ba`, matching `origin/check-equivariance-necessity`  
**Overall verdict:** **NO-GO**

Targeted validator/generator tests pass: **108 passed**. Bash syntax and `git diff --check` also pass. The remaining failures are integration and execution-model defects not exercised by those tests.

## Items 1–8 plus metric-set fix

| Item | Status | Evidence and judgment |
|---|---|---|
| **1 — mandatory evaluator fields, exact types, batch/device, source binding** | **PARTIALLY** | The seventeen evaluator fields are mandatory ([exp11_validate_rows.py:90](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:90)); missing/null fields fail, booleans are excluded from integer fields, batch size is fixed at 64, and `source_sha == sidecar.commit` is enforced ([exp11_validate_rows.py:299](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:299), [exp11_validate_rows.py:337](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:337)). However, “exact types” is not exact: fields registered as `float` explicitly accept integers, so `cfg_scale: 1` and `rotate_deg: 0` pass ([exp11_validate_rows.py:306](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:306)). Device validation accepts any string beginning with `cuda`, including malformed values such as `cudaevil` ([exp11_validate_rows.py:317](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:317)). |
| **Metric-set fix — exact true evaluator emission set** | **CLOSED** | The validator now registers all eleven keys actually emitted by `eval_FLAC`, requires exact set equality in both directions, and validates every emitted value as finite and non-boolean ([exp11_validate_rows.py:74](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:74), [exp11_validate_rows.py:281](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:281)). I independently inspected both real C4 backfill records; each contains exactly those eleven keys. |
| **2 — active-arm checkpoint bound to original launch manifest and canonical run directory** | **PARTIALLY** | Canonical realpath equality and arm/config matching are enforced ([fa_orbit_screen.sbatch:228](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:228), [fa_orbit_screen.sbatch:243](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:243)). The row validator also uses realpath containment rather than substring matching ([exp11_validate_rows.py:355](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:355)). But the manifest is mutable evidence under ignored `outputs_FLAC`, and the gate does not validate its recorded launch commit, `mode INITIAL`, launch UUID/job, rung, training seed/command, P0 manifest hash, or VAE hash. The message even prints “seed 42 recipe” without parsing or checking it ([fa_orbit_screen.sbatch:263](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:263)); those fields exist in the manifest ([launch_manifest.txt:3](/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp11_C8/launch_manifest.txt:3)). This binds the checkpoint to a directory described by the current manifest, not cryptographically to the original audited manifest. |
| **3 — generator recomputes hashes** | **CLOSED**, with an Item 8 availability defect | `validate_exp11_cell()` calls the table contract with `verify_hashes=True` ([gen_model_comparison.py:95](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:95)), and the validator recomputes both config and checkpoint SHA-256 values ([exp11_validate_rows.py:375](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:375)). Missing files fail closed. Under Item 8, however, the sidecar records an ephemeral worktree config path which pruning can remove, so later validation can become permanently blocked. |
| **4 — R3 five-angle contract and two-K outer gate** | **PARTIALLY** | R3 is correctly registered as seed 42 across exactly five rotations ([exp11_validate_rows.py:124](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:124)), and cell validation switches the replication key from seed to rotation ([exp11_validate_rows.py:451](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:451)). The two-K logic is not a gate: rows, including numeric rows, are rendered first; incomplete K coverage merely appends a warning and the generator still writes the table successfully ([gen_model_comparison.py:242](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:242), [gen_model_comparison.py:256](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:256)). One valid K can therefore still publish while its partner is absent or blocked. |
| **5 — no false `validated=table` for one confirmatory row** | **CLOSED** | Futility rows are validated and labelled `validated=futility`; confirmatory rows are explicitly labelled `validated=pending-cell` with the whole-cell validation command ([fa_orbit_screen.sbatch:355](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:355), [fa_orbit_screen.sbatch:373](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:373)). |
| **6 — generator execution behind `main()`** | **CLOSED** | Generation is contained in `main()` and guarded by `if __name__ == "__main__"` ([gen_model_comparison.py:215](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:215), [gen_model_comparison.py:263](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:263)). Importing the module no longer rewrites the table. |
| **7 — preserve numbers while migrating/defer labels explicitly** | **NOT** | `build_header()` contains the intended loud deferral note ([gen_model_comparison.py:150](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:150)), but `main()` never calls it and instead uses a separate hard-coded header ([gen_model_comparison.py:220](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:220)). The preserved artifact still labels historical rows only `fa eval` and contains no deferral disclosure ([model_comparison.md:24](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/model_comparison.md:24)). The tests exercise `build_header()` in isolation, not its integration into `main()`. |
| **8 — worktree-pinned measurement execution** | **NOT** | The intended code/output split is present, and HEAD is read from `CODE_ROOT` ([fa_orbit_screen.sbatch:59](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:59), [fa_orbit_screen.sbatch:167](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:167)). But the pruning policy can delete queued/running jobs’ trees, worktree identity is insufficiently enforced, ignored runtime inputs are absent/unpinned, and later table validation depends on paths to prunable trees. Details follow. |

## Item 8 fresh-eyes review

### Worktree creation, reuse, concurrency and pruning

**Unsafe. `KEEP=3` is not sufficient.**

The helper retains the newest three directories by filesystem mtime and force-removes the rest ([fa_orbit_measure_worktree.sh:23](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:23), [fa_orbit_measure_worktree.sh:57](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:57)). There is no job lease, reference count, Slurm-state check, or lock.

A concrete failure sequence is:

1. Job A is queued with the worktree for SHA A.
2. Three later submissions create SHA B, C and D.
3. D’s helper sees four directories and removes A.
4. Job A eventually starts with `MEASURE_ROOT` pointing to a deleted directory and aborts.

A running job is also unprotected and can have its worktree removed while it still needs imports, configs, or validation code. Concurrent helpers can race on the same `git worktree add`, and different-SHA helpers can interleave enumeration and removal. The `git worktree remove --force || rm -rf` fallback is especially inappropriate without ownership/lease validation ([fa_orbit_measure_worktree.sh:62](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:62)).

The helper also calls `git status` without checking its exit status because it uses `set -uo pipefail`, not `set -e`, and supplies no `|| exit`; a failed status command can be treated as an empty clean result ([fa_orbit_measure_worktree.sh:26](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:26), [fa_orbit_measure_worktree.sh:53](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:53)).

**Required design:** create a unique worktree when the Slurm job starts, under a global `flock`, and remove it through a job-exit trap. Queued jobs should depend only on `EXPECT_SHA`, not on a pre-created directory. If trees are shared by SHA, maintain explicit leases and prune only zero-lease trees. Remove the fixed-count policy and the raw `rm -rf` fallback.

### Binding to worktree HEAD

For a correctly created, untouched detached worktree, `HEAD == EXPECT_SHA` is sound and the evaluator’s post-run `git rev-parse` will resolve through the worktree `.git` file.

The driver does not, however, prove that `MEASURE_ROOT` is such a tree. It accepts any directory containing `.git`, including the mutable main checkout itself ([fa_orbit_screen.sbatch:67](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:67)). It does not require:

- the canonical `.measure_worktrees` location;
- a detached HEAD;
- the same Git common directory as `MAIN_REPO`;
- a top-level path equal to `MEASURE_ROOT`;
- absence of untracked/ignored code;
- a final clean-state check after evaluation.

The cleanliness test explicitly ignores untracked files ([fa_orbit_measure_worktree.sh:54](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_measure_worktree.sh:54)), and the tree is not actually read-only. A reused tree can therefore contain untracked importable code or be mutated after the one-time pre-run status check.

The gate should validate canonical top level, common Git directory, detached state, expected managed path and full clean state, then recheck at completion. Recording the already-gated SHA explicitly in the evaluator record would also avoid relying on a second, post-evaluation Git lookup ([eval_FLAC.py:78](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:78), [eval_FLAC.py:128](/n/fs/gatrdp/codespace/FLAC/eval_FLAC.py:128)).

### Input/output path split

| Surface | Current resolution | Judgment |
|---|---|---|
| Python code, model configs, dataset-config JSONs | Pinned worktree | Correct. |
| Audited C4 backfill manifest | Pinned `EXPDIR` | Correct; its relative checkpoint path is explicitly resolved against `MAIN_REPO` ([fa_orbit_screen.sbatch:197](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:197)). |
| Checkpoints and result/log outputs | Main tree, absolute | Correct in principle; checkpoint hashes are recomputed. |
| Active-arm launch manifest | Main-tree `outputs_FLAC` | Mutable and unpinned; its expected digest and provenance fields are not recorded or checked. |
| Actual AcousticRooms data | Relative path `AcousticRooms` from the worktree | Broken. The dataset config uses that relative path ([acousticroom_unseeneval.json:5](/n/fs/gatrdp/codespace/FLAC/src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json:5)), while `AcousticRooms` is ignored and therefore absent from newly added worktrees ([.gitignore:174](/n/fs/gatrdp/codespace/FLAC/.gitignore:174)). |
| AGREE metric weights | Relative `weights/AGREE/AGREE_fullAR.pt` from the worktree | Broken and unpinned. `weights/` is ignored, but every arm enables FD/retrieval and names that relative checkpoint ([FLAC_AR_BF_C8.json:186](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_BF_C8.json:186)); metric setup immediately loads it ([metric_callback.py:120](/n/fs/gatrdp/codespace/FLAC/src/metrics/metric_callback.py:120)). Existing measurement worktrees contain neither `AcousticRooms` nor `weights`, so a real pinned evaluation cannot complete from pushed HEAD. |
| VAE file | Not directly passed to `eval_FLAC` | Not an evaluation-time path in this driver; the evaluated checkpoint plus load-integrity gate supplies the model state. Its training-lineage hash remains in the launch manifest but is not checked by the screen. |
| HF cache | Absolute `/n/fs/gatrdp/hf_cache` | Accessible but external. The equivalence probe hashes the pinned DINO snapshot; the screen only sets offline mode and does not run that pin gate. The checkpoint load-integrity check limits the risk, but the dependency should still be explicitly gated for a claimed fully pinned run. |

There is a second table-specific path defect: the screen sidecar stores the worktree’s absolute model-config path ([fa_orbit_screen.sbatch:339](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen.sbatch:339)). Once that tree is pruned, generator hash recomputation can no longer open the config. Stable provenance should use the commit plus repository-relative path and hash the Git blob, or copy the exact config into durable evidence beside the metrics.

Finally, `gen_model_comparison.py` itself is not worktree/main split-aware: it derives both its evidence root and output file from its own checkout ([gen_model_comparison.py:13](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:13)). Invoked from a measurement worktree, it sees no ignored `outputs_FLAC` evidence and writes the table into the disposable worktree.

### `.git`-file root walk

**Correct.** Changing the marker test from “`.git` is a directory” to “`.git` exists” properly supports linked worktrees without weakening the upward-walk termination behavior ([exp11_validate_rows.py:34](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:34)). This sub-fix is closed.

## `validate_exp11_cell(repo_root=...)` deviation

**Acceptable.**

The override is used only when explicitly passed, operates on a freshly loaded validator module, and changes both repository containment roots consistently ([gen_model_comparison.py:87](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:87)). Production calls omit it. It is a reasonable fixture seam and does not itself create a production fail-open path.

The associated tests are too narrow, however: they test `check_two_k_coverage()` and `build_header()` independently but do not exercise `main()` to prove that incomplete K coverage suppresses numeric rows or that the deferred header actually renders.

## Required before approval

1. Replace fixed-count pruning with job-start creation plus locking and lease-aware cleanup; never remove a tree referenced by a queued or running job.
2. Enforce that `MEASURE_ROOT` is the expected detached managed worktree for `EXPECT_SHA`, from the correct Git common directory, and check full cleanliness before and after execution.
3. Make AcousticRooms and AGREE resolution explicit and durable; gate their canonical paths and hashes. Run the DINO cache pin in screen jobs if the run is claimed fully pinned.
4. Pin the active-arm launch-manifest digest and validate its seed, launch mode, run identity, recorded commit and relevant artifact hashes.
5. Store model-config provenance independently of an ephemeral worktree path.
6. Make two-K coverage transactional: block both rows or abort without writing when either K is absent/invalid.
7. Integrate `build_header()` into `main()` and update the preserved table with the explicit legacy-loop deferral note without changing its numbers.
8. Enforce truly exact runtime types and a strict CUDA-device grammar.

## Verdict

| Scope from pushed `d273b15` via `MEASURE_ROOT` | Decision | Reason |
|---|---|---|
| **(a) C4 backfill re-runs** | **NO-GO** | The worktree can be pruned while queued/running and lacks the relative AcousticRooms and AGREE assets required by the evaluation. |
| **(b) Arm-screen series** | **NO-GO** | Same worktree lifecycle/input failures, plus incomplete immutable binding to the original launch manifests. |
| **(c) Table pipeline** | **NO-GO** | The two-K check is advisory, label deferral is not rendered, config validation depends on prunable worktree paths, and the generator cannot consume main-tree evidence correctly when run from a measurement worktree. |
