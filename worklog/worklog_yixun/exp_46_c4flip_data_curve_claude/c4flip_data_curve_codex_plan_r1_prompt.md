# Plan review request — exp_46 c4flip_data_curve, plan round 1 of at most 3

You are the Reviewer (OpenAI Codex). This is a read-only review.

**You must not:**
- install, upgrade or modify any Python environment, package, conda env or file;
- write anywhere;
- run pytest, training, evaluation, `sbatch`, `srun` or any GPU or Slurm job.

**Allowed:** reading files and cheap read-only commands (`git show`, `git diff`, `git log`, `git ls-tree`, `git cat-file`,
`sha256sum`, `head`, `grep` on named paths, and `python -c` that only reads). **Never** run `find`, `du` or `ls -R`
across `/n/fs` (shared filesystem); read only the paths named here and files in this repository.

The workdir is the worktree `/n/fs/gatrdp/codespace/exp-46-c4flip-data-curve` (branch `exp-46-c4flip-data-curve`, HEAD
`5d4891cc` = `90d0202f` + one records commit). All exp_46 files are in `worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/`.

## Context

- **The object of review:** `plan_c4flip_data_curve.md` (r0). It is written as a **delta** to exp_45's approved plan r3
  (`worklog/worklog_yixun/exp_45_rf_data_curve_claude/plan_rf_data_curve.md`); read both.
- Yixun's decisions C1–C5 that define the experiment: `c4flip_data_curve_yixun_query.md` (Query 8). They are decided;
  do not re-argue them, but do flag anything in the plan that does not implement them.
- exp_45's kit is in `worklog/worklog_yixun/exp_45_rf_data_curve_claude/exp45_*.py`, tests `src/tests/test_exp45_*.py`
  and `exp45_fixtures.py`. Its code went through three Codex code rounds (reviews `rf_data_curve_codex_code_r{1,2,3}_review.md`);
  the round-3 HIGH-1 / MEDIUM-2 fixes are being made on the exp_45 branch now (`X45f`) and exp_46 is rebased onto them
  before any code. HIGH-1 concerns the RF100 bridge publication gate, which exp_46 removes.
- The training path the plan changes: `src/training/factory.py` (≈ 80–102), `src/training/diffusion.py` (≈ 200–310,
  488–502, 740–785), `src/data/yaw_offset_step.py`, `src/data/yaw_mirror.py`, and the tests listed in plan G5.
- Standing rules: `worklog/experiment_SOP.md`, `worklog/worklog_yixun/announcement/` (01, 04, 05, 07, 08 matter most),
  `CLAUDE.md` at the repo root.

## What to check

Report **every** finding in this single pass (the cap is 3 plan rounds), each with severity BLOCKING / HIGH / MEDIUM /
NIT, plan section or file:line, the concrete failure scenario, and a fix.

1. **Facts G1–G9.** Verify each against its source; flag anything wrong or overstated. In particular G2 (the C4F
   offsets equal exp_40's grid draws and the flip bits equal RF's for the same `(seed, step, rank)`), G3 (D4 exactness
   of quarter turns + mirror), G4 (banner), G5 (is the list of tests that assert the refusal complete? Any other test,
   in any `src/tests/` file, that would break when the two refusals are removed?).
2. **Training-path change (§3.1).** Is removing exactly the two refusals sufficient and safe? Any other code path
   (wrapper constructor, factory, validation / test steps, `_check_yaw_aug_metadata`, the mirror contract check on
   grid-rotated samples, fa_invariant / canonicalize exclusions, evaluation `eval_FLAC.py`) that assumes flip ⇒ no
   `offset_step`? Are the six new tests sufficient to catch a wrong composition (order, generator consumption, grid,
   mirror)? Are E1–E5 the right declared edits, and is E3's claim (the exp_43 kit still refuses `offset_step=32`
   through its own diff rule) true?
3. **Kit fork (§3.2).** Anything in exp_45's kit hard-bound to RF, three runs, fractions only, or the bridge that is
   missing from the delta table? In particular every place where adding a 100 % run (f100) matters: `FRACS`, the
   subset audit binding, `restrict_to_split` static checks, loader-length gate, epoch index, storage, dataset-config
   and split hashing, the argv projection, the collector, the table, the generator contract. Is removing the bridge
   complete and safe (no dangling reader, publication gate or test)?
4. **Comparability.** Does the argv rule make C4F's fraction rows comparable to exp_38's YAW / OrbitRIR fraction rows?
   Is evaluating all four C4F points in tree 46 (no bridge) sound, with comparability to exp_38's rows resting on E0?
   Is the f100 run's data path (full train config) correctly classified?
5. **Isolation and cross-experiment safety.** exp_46 must never write into exp_38's or exp_45's folders, save roots,
   ledgers, owners files or run dirs; the shared `.exp38_card<i>.lock` is the only intentional sharing. Is the
   cross-experiment storage guard (exp_45 ledger) correct and sufficient? Is E4 (bounding exp_45's later-commit audit
   to `merge..X45f`) sound, and does `test_exp46_provenance.py` as specified really confine the `src/` change to the two
   refusals?
6. **Regression comparison (L1).** Is three-way comparison against `3593f34c` / `4a7e3b3a` with the extended node
   policy right when the exp_46 tree also carries exp_45's never-L1-verified kit and tests?
7. **Ladder, acceptance, schedule, disclosures, storage, decisions M1–M4.** Missing rungs or criteria (announcements 01,
   05, 07, 08), `--time` rule, concurrency of 8 L40, disk (94 % full), the push of exp_45's unpushed commits as
   ancestors (M4). Give a one-line view on each of M1–M4.

## Output

1. Findings, most severe first (severity, location, failure scenario, fix).
2. One line per fact G1–G9: CONFIRMED / WRONG / OVERSTATED.
3. One line per decision M1–M4.
4. A one-line overall verdict: APPROVE, APPROVE-WITH-CHANGES or REVISE.
