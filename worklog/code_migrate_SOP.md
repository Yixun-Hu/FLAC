# Code-Migration SOP (portable) — DRAFT v2 for owner review; no migration starts until approved

A standard operating procedure for migrating research code that has already produced validated
results into a clean, reviewable package **without changing behavior**. Companion to
`worklog/experiment_SOP.md`, which governs *new* experiments: a migration is done when the
migrated code **numerically reproduces the recorded results**; the reproduction runs themselves
are then executed as a normal experiment under `experiment_SOP.md`.

Shared machinery is inherited from `experiment_SOP.md` and not duplicated here: the three-model
role split and reviewer-reciprocity rules (§Roles), the worklog entry template, the review-file
identity header, and the general commit discipline (SHAs logged everywhere).

## Success criterion (what "migration correct" means)

The migrated package regenerates the pinned reference results from the pinned checkpoints and
episode files — per-item agreement within stated tolerances — **and** every difference between
source and target code is declared, justified, and reviewable. "The code looks equivalent" is
never sufficient; only gates green on numbers count.

## Roles

| Role | Who | Migration duty |
|---|---|---|
| Planner | main session (Claude Fable 5) | writes the migration plan (source inventory → target map → gate numbers); judges gates; for **every migration commit** writes the approval recommendation to the owner (§Unit of migration, item 5); reports at phase boundaries |
| Coder | Opus 5 max subagent | executes migration commits exactly per the approved plan |
| Reviewer | OpenAI Codex, strongest model at `xhigh` — currently `gpt-5.6-sol` (per Yixun 2026-07-10; supersedes `gpt-5.5`; requires codex-cli ≥ 0.144) (per reciprocity rule) | reviews the migration PLAN before owner sign-off; reviews every migration round's diffs before the next round opens |
| **Owner (Yixun)** | — | approves the plan; **approves every migration commit** (based on the Planner's per-commit analysis); may halt or amend at any point |

## Unit of migration: one function per commit (the core rule)

> **Prime directive: the owner (Yixun) must be able to understand every migration commit.**
> Every item below is written for a human reader first — plain language, no unexplained
> jargon, evidence shown not asserted — and every commit ends by stating the next step, so
> the owner always knows both *what just happened* and *what comes next*.

Migration proceeds **function by function** (a single function, or one minimal coherent unit
such as a small class with its methods). **Every migration commit is < 200 changed lines
including its tests**, and MUST ship with six items:

1. **Explanation** — what this function does (its role in the pipeline, inputs/outputs), in
   the commit body and the ledger row.
2. **Sameness classification** — is the migrated code *exactly* the same as before, or the
   *same in function*? One of:
   - `exact` — logic byte-identical; only T1 (imports/paths) and T3 (docs/typing) applied;
   - `equivalent` — behavior identical, text differs (e.g. T2 flag-stripping, function split);
     the claim of equivalence is exactly what item 3 must prove;
   - `delta` — intentional behavior change (rare; needs one-sentence justification and
     explicit owner attention in item 5).
3. **Test functions proving same behavior** — characterization test(s), pytest: pin the SOURCE
   function's behavior on fixtures (run against the source first — green), then the same test
   must pass against the TARGET. Same-commit as the migrated function (never after). Where the
   source cannot run standalone, characterize at the nearest testable boundary and record that
   limitation in the ledger row. Tests are permanent regression assets in the package's
   `tests/` folder.
4. **Test results** — the actual run output (pass/fail, key numbers), recorded at commit time.
5. **Analysis & recommendation** — the Planner's (Fable 5) judgment: does the evidence support
   this commit? Any residual risk? Ending in an explicit recommendation to the owner:
   **approve / revise** — the owner decides.
6. **Next step** — one or two plain sentences: which function/commit comes next and why, so
   the owner is never surprised by the following commit.

**Where the six items live** (per owner's design):

| Item | Artifact |
|---|---|
| 1 | **`<name>_explanation.md`** (append-only, keyed by commit; announcements #13/#14) — each entry: (1) changed-line counts with per-file breakdown, (2) count + names of ALL functions in the commit (migrated/test/glue, labeled), (3) per-function explanation; + commit message body |
| 2 | commit message body + `MIGRATION_LEDGER.md` row |
| 3 | the test file(s) in the commit itself |
| 4 (how to run) | `<name>_command.md` — the exact test command(s), appended **at commit time** |
| 4 (output) | `<name>_results.md` — the test results, appended at commit time |
| 5 | `<name>_analysis.md` — the Planner's per-commit approval recommendation to the owner |
| 6 | closes the `<name>_analysis.md` entry (and the `_worklog.md` entry's **Next** field) |

All three .md files are **append-only running logs keyed by commit SHA + function name** —
one entry per migration commit, written when the commit is made, never retroactively.

**Approval loop**: after each commit's five items are in place, the owner approves or rejects
(recorded in `<name>_worklog.md`). Default cadence is **synchronous** — the next function's
commit waits for the owner's approval; the owner may explicitly grant batch pre-approval for a
scoped set (e.g. "the rest of this phase") to speed things up. A rejected commit is fixed or
reverted before anything builds on it.

## Bookkeeping

One folder per migration: `worklog/mig_<NN>_<name>_claude/` (never `exp_`), containing:

1. `<name>_yixun_query.md` — the owner's driving queries, verbatim + summary.
2. `plan_<name>.md` — Planner's migration plan: pinned source commit, full source-file
   inventory with sha256, the **function-level** source→target map (each function's target
   module and planned commit), the allowed-transform table, the stripped-feature list, phase
   order, and the **gate table with exact expected numbers and tolerances**.
3. `<name>_codex_plan_review.md` — Reviewer's plan review (before owner approval).
4. **Owner approval of the plan** — recorded verbatim in `<name>_worklog.md`. Nothing is
   migrated before it.
5. `MIGRATION_LEDGER.md` — one row per migration commit: function(s), source file@pin +
   sha256, target file, sameness classification (`exact`/`equivalent`/`delta` + delta IDs),
   test file, commit SHA, phase, gate coverage.
6. `diffs/<NN>_<function>.diff` — per-commit unified diff (source slice → target). **The
   primary review object** for the owner and the Reviewer.
7. `<name>_codex_code_<marker>_review.md` — per-round Reviewer review; a **round** = one
   coherent group of function-commits (typically one source module). Same round-closure rule
   as experiment_SOP: blocking findings fixed and re-verified before the next round opens; an
   integrative review (marker `full`) before the first expensive gate (G5).
8. `<name>_worklog.md` — append-only notebook (experiment_SOP entry template): every commit,
   every owner decision, every gate run, every failure triage.
9. `<name>_command.md` — per-commit test commands + gate/verification commands, at launch time.
9b. `<name>_explanation.md` — per-commit plain-language function explanations (item ①),
    append-only, at commit time (announcement #13).
9c. `<name>_params_set_up.md` — per-commit DATA-config and NETWORK-config parameter setup
    (pinned inputs, seeds, modes, expected counts/hashes, tolerances, env), append-only at
    commit time, aligned 1:1 with the same commit's `_command.md` and `_results.md` entries
    (announcement #19).
10. `<name>_results.md` — per-commit test results + gate outputs, appended as they run.
11. `<name>_analysis.md` — per-commit Planner recommendations (approve/revise) + the final
    migration judgment: gates summary, declared deltas, residual risks, handoff statement.
12. `commits_<name>.md` — SHA + one-liner for every migration commit.

## Reviewer briefing (migration) — load Codex with context before it judges

Every Codex invocation (plan review AND each code round review) must direct it to read first:
1. `worklog/code_migrate_SOP.md` (this file), `worklog/experiment_SOP.md`, and all
   `worklog/announcement/` directives;
2. `plan_<name>.md`, `MIGRATION_LEDGER.md`, and `<name>_worklog.md` (decisions and amendments
   so far, including owner rejections);
3. the pinned SOURCE file(s) for this round, the migrated TARGET file(s), the round's
   `diffs/*.diff`, its tests, and their recorded results;
4. a one-paragraph statement of what this round was tasked to migrate and what is explicitly
   out of scope for the round.
A reviewer without this context produces generic reviews, flags out-of-scope "gaps", and
misses violations of migration-specific decisions. Review files carry the standard identity
header (model, version, invocation, date) per experiment_SOP.

## Per-commit procedure M1–M8 (function granularity)

- **M1 Pin & hash** — read the source at the pinned commit only; record `sha256(source file)`.
- **M2 Map** — the function's target module per the approved plan.
- **M3 Transform** — allowed without being a delta: **T1** import/path rewrites; **T2**
  removal of features the plan explicitly strips (dead flags and their branches); **T3**
  docstrings, type hints, provenance header; **T4** owner-directed renames (announcement #11:
  no cvae/CVAESB prefixes inside the package) — valid ONLY when (a) an owner directive
  requires it, (b) the step plan carries the old→new naming map, (c) the ledger row repeats
  the mapping, (d) external pinned artifacts are never renamed (package-GENERATED future
  artifacts use new names; pinned historical inputs/results keep old names). Anything else ⇒
  classification `delta`.
- **M4 Provenance header** — every target file opens with (functions accumulate as they land):
  ```python
  # Migrated from <source path> @ <pin>
  # Functions: <fn> (<exact|equivalent|delta:D1>, commit <sha>), ...
  # Deltas: D1 <one line: what + why>          # omit if none
  ```
- **M5 Diff artifact** — `diffs/<NN>_<function>.diff` = `diff -u <source slice> <target>`.
- **M6 Characterization test** — item 3 above; green on source first, then green on target.
- **M7 Commit & push** — `mig: <function> <- <source>@<pin> (<exact|equivalent|delta>)`,
  < 200 lines including tests, pushed immediately; items 1–5 recorded in the same action.
- **M8 Ledger row** — same action as the commit.

**Non-code artifacts** (data lists, episode JSONs, configs): copied byte-identical with
`sha256(source) == sha256(target)` recorded in the ledger; configs whose paths must be
re-rooted are treated as code (diff + one delta per changed path).

## Verification gates (cheapest-first; the migration's validation ladder)

Defined generically here; `plan_<name>.md` binds each gate to exact artifacts, expected
numbers, and tolerances. A red gate stops the line: fix the migration, never the expected
numbers; root-cause goes in the ledger row and `_worklog.md`.

- **G1 Static + tests** — py_compile every module; full pytest (all characterization tests) green.
- **G2 Data-artifact identity** — rebuilt lists/pools/splits identical to the pinned ones
  (set-equal or byte-identical as the plan states).
- **G3 Property tests** — the invariants the method claims (e.g. symmetry/equivariance) hold
  on real data within stated tolerance.
- **G4 Checkpoint parity (cheap slice)** — the pinned checkpoint loads strict; a small pinned
  eval slice reproduces its recorded numbers within tolerance.
- **G5 Full-cell parity** — one full pinned evaluation cell reproduces the recorded aggregate
  AND per-item records within tolerance (per-item deviation is the real test; aggregates can
  mask compensating errors). Requires the Reviewer's `full` integrative review first.
- **G6 Downstream-verdict parity** — the statistical comparison recomputed from the new
  per-item output reproduces the recorded verdict.

G4–G6 are verification reruns of already-recorded results using existing checkpoints — they
are not new experiments and do not require experiment_SOP scaffolding.

## Phases and cadence

Migrate in dependency order, one phase at a time (typical: data → models → training → eval →
baseline wrappers → artifacts → tests/docs). Within a phase: function-commits under the
per-commit approval loop; Codex round reviews per source module. A phase ends with: its gates
green → ledger complete → all round reviews closed → **phase report to the owner**; the next
phase starts only after that report. The owner may halt or amend at any boundary.

## Running & environment discipline

Inherited from `experiment_SOP.md` §Running & failure discipline verbatim, plus:
- The source checkout/worktree is **read-only** for the entire migration.
- Legacy copies of the source tree in the working checkout (kept alive for still-running
  experiments) are neither sources nor targets — never modify them.
- Never touch running trainings/watchers; gate runs share GPUs politely (small batch).

## Handoff

Migration closes with `_analysis.md`'s final judgment + owner sign-off. The full reproduction
of the migrated method's results (training from scratch through final tables) is then
scaffolded as `worklog/exp_<NN>_<name>_repro_claude/` under `experiment_SOP.md`, using
**only** the migrated package; its baseline-calibration step compares against the recorded
numbers of the original code.
