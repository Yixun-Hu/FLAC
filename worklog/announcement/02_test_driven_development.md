# 02 — Test-driven development for all new code

**Logged:** 2026-07-04 (from Yixun)

## Original instruction (verbatim)

> You should use test-driven development: https://en.wikipedia.org/wiki/Test-driven_development. First, determine each small functions test function, then use this test functions to develop corresponding functions, splitted into small commits (same requirements in the @worklog/SOP.md).

## Rule

- Every new function is developed test-first: define the test function(s) for each small unit BEFORE implementing it; watch the test fail (red), implement the minimal code to pass (green), refactor with tests green.
- Tests are pytest files in a dedicated tests folder; the folder's location must be confirmed with Yixun before the first tests land. **For this FLAC repo: `src/tests/`** (decided 2026-07-04). Tests are committed with (or immediately before) the implementation they drive — never after.
- Each red→green cycle is one small commit (composes with the < 200-line commit rule in `worklog/SOP.md`).
- Experiment plans (`plan_<exp name>.md`) must enumerate the per-function test list before implementation begins.
- Existing tests are permanent regression assets: they must keep passing in every later experiment (validation-ladder rung between static checks and smoke runs).

## Why

TDD forces each function's contract to be stated before the implementation exists, catches regressions when later experiments touch shared code (e.g. the yaw-symmetry utilities), and produces the small, verifiable commits the SOP's reliability discipline requires.

## Note on precedence

The repo's CLAUDE.md previously noted "no test suite — don't invent one" (describing the upstream release state). This announcement supersedes that note for all new development in this fork: new code gets tests; upstream release code is not retroactively test-covered unless an experiment touches it.
