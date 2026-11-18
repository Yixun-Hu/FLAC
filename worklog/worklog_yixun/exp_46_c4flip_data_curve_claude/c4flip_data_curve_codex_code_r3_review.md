**Reviewer:** OpenAI Codex (codex-cli 0.166.0, `codex exec`, read-only sandbox, model `gpt-6.1-sol`, reasoning `max`, workdir `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve`) · **Date:** 2026-11-18

Reviewed `fa6b8eee` read-only. No tests, jobs, dataset reads, or environment changes were performed.

- **R2-1 — PARTIAL:** ordinary child aliases are guarded; the matrix detects the original failures. Normalizing before validation still permits a `symlink/..` bypass, described below.
- **R2-2 — RESOLVED:** both regenerations use `-B`; the subprocess test includes a meaningful control without it.
- **R2-3 — RESOLVED:** init-digest checks and writes the same absolute destination; the non-root-cwd regression detects the original bug.
- **R2-4 — RESOLVED:** rendering compares supplied values with source aggregates; finite-value mutation tests detect the missing comparison.
- **R2-5 — RESOLVED:** verified exp_38 import and dependency checks precede materialisation; the ordering assertions reject the previous sheet.
- **R2-6 — RESOLVED:** the four registered runs have executable registry/SMOKE/FULL dry-runs before freeze. These paths submit nothing. Failure propagation remains a separate sheet issue.
- **R2-7 — RESOLVED:** publication-worktree creation and HEAD/branch/upstream/cleanliness checks are explicit; the tests detect their absence.
- **R2-8 — PARTIAL:** exclusive staging and `RENAME_NOREPLACE` are sound; the fallback remains racy, and its test misses the final race window.
- **R2-9 — RESOLVED:** sealed-run abandonment refuses, and release preserves the completed run; the cancellation regression detects removal of that guard.
- **F2 — PARTIAL:** the init-digest anchoring defect is fixed, but auxiliary writers remain affected by the shared destination-validation bypass.
- **F8 — RESOLVED:** the actual rendering boundary rejects finite source disagreements.
- **F10 — PARTIAL:** both sheets exist, but failure handling, asynchronous ladder barriers, and initial NAS-directory provisioning remain incomplete.

**F6/E7 pre-review:** the patch is **correct for that test and every existing Lightning reference**, but **insufficient as the sole full-suite repair**.

Monkeypatch records the incoming attribute on the shared callable. Fabric and PyTorch Lightning re-export that callable, and existing decorated banners consult its rank when called; teardown therefore restores the value they read. No separate restoration of each module alias is needed. [Fabric source](https://raw.githubusercontent.com/Lightning-AI/lightning/2.1.0/src/lightning/fabric/utilities/rank_zero.py), [PyTorch Lightning source](https://raw.githubusercontent.com/Lightning-AI/lightning/2.1.0/src/lightning/pytorch/utilities/rank_zero.py), [decorator source](https://raw.githubusercontent.com/Lightning-AI/utilities/v0.15.2/src/lightning_utilities/core/rank_zero.py).

The required assertions are:

- **Provenance:** compare the complete file against X45f plus exactly the supplied four-line insertion at the unique specified anchor. Require unchanged mode and no other additions, deletions, assertion changes, or node renames. Cover committed and tracked working-tree contents; merely allowing changes anywhere in the named function is insufficient.
- **L1 registration:** add the exact sampler node to `DECLARED_EDITED`. Require it to be present and **passed**, including when absent from both parents. Reject failure, error, skip, or an inherited-failure exception.
- **Regression coverage:** exercise teardown and check the shared references, including the banner’s reference. The check must fail without E7. Also address the separate exp_38 polluter below before claiming whole-suite isolation.

New findings, most severe first:

1. **HIGH — Validation and mutation can address different paths.** [exp46_registry.py:1090](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_registry.py:1090), [exp46_registry.py:1121](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_registry.py:1121).  
   `destination_problems()` collapses `..` with `abspath()`, but writers use the original argument. With the A6000 sheet’s required `AcousticRooms` symlink, a comparator output such as `AcousticRooms/../checkpoints/exp38_data_efficiency/report.json` is checked as `<campaign>/checkpoints/…` yet opened inside exp_38’s NAS checkpoint namespace. `O_NOFOLLOW` does not protect intermediate components. **Fix:** bind every mutation to the exact absolute path validated, and reject unsafe traversal through symlinks. Cover `symlink/..` through the actual writers; the current alias matrix omits it.

2. **MEDIUM — First-time E0 snapshot preparation always refuses the shared container.** [exp46_eval.py:237](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_eval.py:237).  
   The guarded `makedirs(outputs_FLAC)` rejects that directory because it contains registered protected exp_38/exp_45 farm namespaces—even when the directory already exists. Thus a clean campaign cannot create its first evaluation snapshot. The E0 fixture replaces `eval_snapshot()`, hiding this path. **Fix:** reuse the verified existing container or provide a narrowly scoped container-directory check without relaxing protections on foreign descendants. Add a positive first-snapshot case using the real guards.

3. **MEDIUM — E7 leaves another whole-suite rank polluter untouched.** [test_exp38_sampler.py:100](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/src/tests/test_exp38_sampler.py:100).  
   This original test also leaves the shared rank at 1. E7 and exp_46’s cleanup correctly preserve an incoming 1; they cannot remove that pollution. A subsequent banner assertion is still suppressed. The exp_43 banner node has no counterpart in the exp_38 parent, so that cross-lineage failure cannot simply be inherited. **Fix:** obtain D1 coverage for an analogous, precisely declared cleanup of this mutator, or another approved isolation mechanism. Add a regression covering the upstream sampler followed by a real banner assertion.

4. **MEDIUM — The NAS fallback is not atomic no-clobber publication.** [exp46_nas_publish.py:66](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_nas_publish.py:66).  
   After the final `lexists()` check, another publisher can create an empty destination directory; `os.rename()` then replaces it successfully. The tests create the competitor before that final check. The sheet targets the correct NAS path, but unsupported-filesystem fallback remains unsafe. **Fix:** refuse unsupported no-replace operations or implement an equivalent serialized publication protocol. Test an unsupported `renameat2` result and creation immediately before the fallback rename; correct the mutation-audit claim too.

5. **MEDIUM — A6000’s background evaluation failure is discarded.** [A6000 sheet:64](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/c4flip_data_curve_command.md:64).  
   `set -e` does not propagate the background K8 process’s failure. Bare `wait` returns success after waiting for all children, so a successful K1 lane permits continuation despite K8 failing. Collection subsequently protects incomplete publication, but the stated failure handling is false. **Fix:** capture and explicitly check each lane’s PID, and handle the remaining child when either lane fails. Cover the shell behavior with failing stand-ins.

6. **MEDIUM — The neuronic sheet orders submissions, not successful ladder completion.** [Neuronic sheet:71](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/c4flip_data_curve_neuronic_commands.md:71), [Neuronic sheet:87](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/c4flip_data_curve_neuronic_commands.md:87).  
   L2c and L3 use asynchronous `sbatch`; execution can reach freeze while they remain queued or have failed. The shell also continues after a failed comparison or dry-run. The stated D2 order is correct, but textual ordering tests do not enforce its gates. **Fix:** add explicit successful-completion barriers and checked failure exits before advancing or freezing; ensure a failed dry-run prevents the push.

7. **MEDIUM — Fresh NAS output namespaces are never provisioned.** [exp46_nas_publish.py:38](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_nas_publish.py:38), [exp46_eval.py:954](/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve/worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_eval.py:954).  
   Staging requires the exp_46 checkpoint root to exist, and result evaluation requires the exp_46 dump root. Neither sheet creates these new namespaces or establishes their existence as a prerequisite. A first campaign stops at publication and again at dump storage admission. **Fix:** document and test isolated provisioning of both directories after verifying their parent NAS mount; preserve refusal when the mount is missing.

The FULL argv retains the registered substitutions. I found no Git operation that modifies `main`.

Overall verdict: **REVISE**.