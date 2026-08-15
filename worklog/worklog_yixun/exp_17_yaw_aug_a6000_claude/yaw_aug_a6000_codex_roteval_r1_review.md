**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-15 · **Round:** rotation-grid r1 / completion-audit r2

Reviewed at HEAD `ae74320`. Static review only.

---

Overall verdict: **REQUEST-CHANGES. Do not launch the 12-hour grid.** The largest issue is that every generated evaluation command uses the wrong conditioner autocast protocol.

| File | Verdict |
|---|---|
| `src/tools/exp17_rotation_grid.py` | REQUEST-CHANGES |
| `src/tests/test_exp17_rotation_grid.py` | REQUEST-CHANGES |
| `src/tools/exp17_full_audit.py` | REQUEST-CHANGES |
| `src/tests/test_exp17_full_completion_audit.py` | REQUEST-CHANGES |
| `yaw_aug_a6000_roteval_run.sh` | REQUEST-CHANGES |

### `src/tools/exp17_rotation_grid.py` — REQUEST-CHANGES

- **Blocking — all 128 commands use the wrong autocast protocol.** [exp17_rotation_grid.py:96](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:96) omits `--cond-autocast bf16`. Therefore `eval_FLAC.py` uses CUDA’s fp16 default, including for vanilla conditioning, rather than the registered bf16 protocol. See [eval_FLAC.py:976](/home/yixunhu/codespace/FLAC/eval_FLAC.py:976), [plan_yaw_aug_a6000.md:99](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/plan_yaw_aug_a6000.md:99), and [model_comparison.md:4](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/model_comparison.md:4). This would silently produce 128 protocol-mismatched numbers.

- **Blocking — `pending_cells()` can skip an unfinished or wrong cell.** [exp17_rotation_grid.py:117](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:117) accepts any non-empty recursive `*<cell_name>*.json`. It neither computes the exact `eval_FLAC` filename nor parses the record. A rot90 eval-name accidentally run with `--rotate-deg 0` produces no trailing `_rot90`, but still matches and is considered complete. `{}`, truncated JSON, stale fp16 output, online-weight output, or `n_samples != 6337` also counts.

- **Non-blocking — other registered defaults remain implicit.** [exp17_rotation_grid.py:96](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:96) also omits the declared `--frame-avg-max-fwd-samples 64` and batch-size provenance. They are currently defaulted/inert for vanilla, but contradict the explicit-protocol contract.

- **Non-blocking — checkpoint labels are trusted.** [exp17_rotation_grid.py:58](/home/yixunhu/codespace/FLAC/src/tools/exp17_rotation_grid.py:58) validates dictionary keys but not path uniqueness or agreement between the key and filename step.

Verified correct:

- The planner emits exactly 16 × 2 × 4 unique tuples.
- K=1 and K=8 point to the correct published configs and their current `max_context` values are 1 and 8.
- `--cond-method vanilla`, fixed rotation, and omission of `--rotate-seed` are correct and accepted.
- Final metric paths are unique. Rot0 has no evaluator suffix but its eval-name contains `rot0`; nonzero cells gain `_rot90`, `_rot180`, or `_rot270`. No two planned cells overwrite each other.
- A normal, correctly named, non-empty completed output is not rerun.

### `src/tests/test_exp17_rotation_grid.py` — REQUEST-CHANGES

- **Blocking — the registered bf16 protocol is untested.** [test_exp17_rotation_grid.py:129](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_grid.py:129) checks cfg scale, diffusion steps, and seed, but not `--cond-autocast bf16`.

- **Blocking — the tests explicitly bless invalid completion evidence.** [test_exp17_rotation_grid.py:163](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_grid.py:163) writes `{}` and requires it to suppress the cell. `{}` is not a completed `eval_FLAC` metrics record.

- **Non-blocking — collision coverage stops at `cell_name`.** [test_exp17_rotation_grid.py:49](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_grid.py:49) never tests the complete checkpoint basename plus evaluator-added rotation suffix.

- **Non-blocking — full-split/K checks are lexical.** [test_exp17_rotation_grid.py:83](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_rotation_grid.py:83) checks filename substrings rather than parsing `max_context`, split path, and full-set identity.

### `src/tools/exp17_full_audit.py` — REQUEST-CHANGES

- **Blocking — it likely rejects a valid completed run.** `_lines()` plus exact membership at [exp17_full_audit.py:62](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:62) and [exp17_full_audit.py:105](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:105) require the Lightning terminator to occupy a line alone. The checked-in real smoke log has it appended directly to the final tqdm record at [smoke log:114](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_2026-08-15_14-14-02_exp17_YAWAUG_smoke_train.log:114). A 40k mid-epoch stop should have the same framing, so gate 2 can reject a valid run.

- **Blocking — the stale-banner defect remains across resumed/appended invocations.** [exp17_full_audit.py:78](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:78) still treats the entire log as one unordered evidence bag. Invocation 1 can supply the exact banner, topology, and losses while invocation 2—without augmentation or with one rank—supplies the endpoint. Whole-line matching fixes quoted/negated substrings, but does not bind evidence to the completing invocation.

- **Blocking — the log is not actually bound to the audited worktree/directory.** [exp17_full_audit.py:91](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:91) uses suffix matching and accepts any matching occurrence. The same relative `outputs_FLAC/exp17_YAWAUG` in another worktree passes. An argv containing `--save-dir expected --save-dir actual-other` also passes because only the first occurrence is captured, while argparse uses the later value.

- **Blocking — checkpoint identity is not exact.** [exp17_full_audit.py:113](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:113) collapses names into a step set. Duplicate same-step checkpoints and unmatched extras such as `junk.ckpt` are ignored. Malformed names such as `optimizer_step=2500foo.ckpt` can also satisfy a cadence step. Ordinary multi-epoch names parse correctly; repeated-step artifacts from resumes do not.

- **Blocking — loss parsing still accepts path/W&B counterfeits.** `_LOSS` at [exp17_full_audit.py:57](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:57) has no contextual boundary. A benign path ending `/pretrain/loss=inf` is treated as a non-finite training loss; fifty `/pretrain/loss=0.5` lines can satisfy the observation floor without real loss evidence.

- **Blocking — short valid resumes are rejected.** [exp17_full_audit.py:141](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:141) requires 50 observations in this log. A legitimate final resume covering fewer than 50 logged observations fails; concatenating old and new logs instead creates the stale-evidence problem above.

- **Non-blocking — live symlinks count as regular checkpoints.** [exp17_full_audit.py:173](/home/yixunhu/codespace/FLAC/src/tools/exp17_full_audit.py:173) uses `Path.is_file()`, which follows non-dangling symlinks. Directories, dangling links, and zero-byte files are correctly excluded.

Genuinely fixed: quoted/negated marker substrings no longer match; missing and non-finite `train/loss=` values are rejected; `rc=None` is rejected and CLI `--rc` is required. The exact-cadence, stale-evidence, and directory-binding blockers are not fully fixed.

### `src/tests/test_exp17_full_completion_audit.py` — REQUEST-CHANGES

- **Blocking — the happy fixture has unrealistic termination framing.** [test_exp17_full_completion_audit.py:52](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_full_completion_audit.py:52) places the termination marker on its own line, hiding the real tqdm/Lightning behavior.

- **Blocking — stale resumed-invocation evidence is untested.** [test_exp17_full_completion_audit.py:117](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_full_completion_audit.py:117) covers quoted and negated strings but not two appended invocation segments combining stale banner/topology/loss evidence with a later endpoint.

- **Blocking — cadence and save-dir adversaries are missing.** [test_exp17_full_completion_audit.py:139](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_full_completion_audit.py:139) omits duplicate steps, unparseable extras, and malformed filenames; [test_exp17_full_completion_audit.py:209](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_full_completion_audit.py:209) omits same-relative-path/different-worktree and duplicate-flag cases.

- **Blocking — the path tests do not exercise the actual loss-parser boundary.** [test_exp17_full_completion_audit.py:180](/home/yixunhu/codespace/FLAC/src/tests/test_exp17_full_completion_audit.py:180) uses paths containing `nan`/`inf`, but none contains the parser-visible substring `train/loss=`.

- **Non-blocking — CLI/filesystem and 49-versus-50 boundaries are untested.** The tests call only the pure function, so required CLI `--rc`, zero-byte/dangling filtering, and the precise loss floor are not regression-pinned.

### `yaw_aug_a6000_roteval_run.sh` — REQUEST-CHANGES

- **Blocking — it executes the wrong fp16-default commands.** [runner:79](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:79) consumes the planner output unchanged, so all scheduled cells inherit the missing bf16 flag.

- **Blocking — gate 2 is unsound in both directions.** [runner:56](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:56) hard-codes `--rc 0` rather than deriving the actual foreign training exit status, defeating the rev-2 rc requirement. It also inherits the valid-termination false rejection described above.

- **Blocking — the “all writes stay local” guarantee is not enforced.** `EXPDIR` and `FARM` are cwd-relative and `mkdir -p` accepts an existing directory symlink at [runner:25](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:25) and [runner:36](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:36). Invoking from the foreign worktree or having `FARM` point there makes links and metrics land in the foreign namespace. With the intended cwd and a real local farm directory, the lexical dirname behavior is safe.

- **Blocking — stale farm entries can select the wrong checkpoint.** [runner:64](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:64) overlays links without refusing unexpected entries. `N` increments even if `ln` fails, and the dictionary at [runner:85](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:85) silently collapses two filenames carrying the same step, potentially choosing a stale target.

- **Blocking — no artifact admission or manifest validation exists.** Final success at [runner:114](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:114) does not check embedded step/config, checkpoint or config hashes, EMA use, training/evaluation source SHAs, `n_samples == 6337`, protocol fields, JSON completeness, or metric finiteness. `eval_FLAC.py` can fall back to online weights and merely record that fact at [eval_FLAC.py:758](/home/yixunhu/codespace/FLAC/eval_FLAC.py:758). This violates the registered validation requirement at [plan_yaw_aug_a6000.md:125](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/plan_yaw_aug_a6000.md:125).

- **Blocking — evaluator failures are discarded.** Both waits at [runner:110](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:110) use no PIDs. A no-argument Bash `wait` returns zero after waiting, irrespective of child failures, per the [GNU Bash manual](https://www.gnu.org/s/bash/manual/bash.html). Missing outputs are only reported after the whole queue; malformed non-empty outputs can still produce a false “all complete.”

- **Blocking — concurrent invocations are unsafe.** There is no lock around the fixed farm, queue, cell logs, or metric paths at [runner:79](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:79). Two starts can schedule duplicate cells, place two processes on each GPU, and race writes to the same JSONs.

- **Non-blocking — gate 1 is narrower than advertised.** [runner:45](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_roteval_run.sh:45) checks one overridable PID, not the launcher/rank-1 process tree. Also, the suggested `TRAIN_PID=` escape does not work because `${TRAIN_PID:-1012326}` restores the default. With the intended current PID alive, the gate does refuse correctly.

Within one runner instance, the GPU scheduling itself is correct: jobs alternate physical GPUs 0 and 1, each sees one device through `CUDA_VISIBLE_DEVICES`, and every pair is barriered before the next pair. On partial failure, however, the script continues; a missing JSON yields `INCOMPLETE` only after the queue, while a non-empty bad JSON can yield false success.

No tests, runner scripts, training, package operations, environment changes, or GPU commands were performed, and no files were modified.
