# exp_36 yaw_c4_spread — commit ledger (branch `exp-36-yaw-c4-spread`, cut from H35 `5b065801`)

Line counts by `exp35_count_code_lines.py` (added + removed lines that are code: not blank, not comment-only, not a
docstring). "code" = outside `src/tests/`; "tests" = `src/tests/`; "data" = `*.json / *.md / *.log / *.sha256`.
RED commits hold the tests (and the RED log), GREEN commits the implementation (and the GREEN log).

| commit | cycle | what | code +/- | tests +/- | data +/- |
|---|---|---|---|---|---|
| `e68b7a04` | records | the exp_36 records imported byte-identically from `localization-exp` `de309efc` (query, plan r0, Codex plan review, Coder brief r1, notebook) + the worktree asset-link log | 0 | 0 | +233 |
| `42d94afc` | 1 RED | YAW-C4S block tests (25 rows, orders 346-370, prio 18, replay first); H35 render 440eb1be + exp_25 render 1073ef64 pinned; H35-count tests scoped | 0 | +122 / -15 | +224 |
| `c4cbebb6` | 1 GREEN | generator: YAW-C4S (explicit names), manifest regenerated (371 rows); `DEFAULT_EXCLUDED_BLOCKS` in `select_cells` (Codex B2, H1) | +23 / -6 | 0 | +109 / -4 |
| `2991a846` | 2 RED | `uses_no_flash` vs `is_exp35`; exp_36 name validator; test helper picks receipt 2 by `uses_no_flash` (Codex B1) | 0 | +91 / -2 | +283 |
| `0818025b` | 2 GREEN | collector: `uses_no_flash` drives no_flash / receipt 2 / backend log / `build_receipt` / `--count-no-flash`; `exp36_name_reasons` | +24 / -9 | 0 | +639 |
| `34b8de25` | 3 RED | `--exp36-replay-check` tests (archived digests first, cross-worktree verification, type-strict equality, exit 0/1/2, verdict) | 0 | +167 | +509 |
| `6721d8c2` | 3 GREEN | `exp36_reference`, `exp36_replay_check`, CLI `--exp36-replay-check / --verdict-out` (Codex B3, H2); one unreachable test case replaced | +127 | +5 / -3 | +481 |
| `ecbaaa63` | 4 RED | `--exp36-publish` tests (Codex B4, H4) | 0 | +145 | +410 |
| `06eeaffa` | 4 GREEN | `exp36_publish` + CLI: HEAD gates, pinned plain destination, verdict re-computed, 48-path preflight, temp + fsync + no-replace link + re-hash, manifest, partial + resume | +134 | 0 | +419 |
| `cc6ded6d` | 5 RED | the fail-closed chain + the runner binding it into the no-flash relevant tree | 0 | +184 | +396 |
| `6a58a160` | 5 GREEN | `yaw_c4_spread_chain.sh` (replay -> check -> the 24 only on an exact 0); runner `RELEVANT += chain` (Codex B4, H5, H6) | +86 / -1 | 0 | +32 |
| `87091900` | 6 RED | a crash is never a mismatch / pass | 0 | +20 / -2 | +220 |
| `ceed2e8c` | 6 GREEN | crashed check -> 2, crashed publish -> 3; the chain needs rc AND verdict to agree | +25 / -13 | 0 | +84 |
| `65f9fcd4` | 7 RED | amendment A1: RPL-S42 + OFF-S42 tests; test-side fix of a cycle-5 defect (the shared sparse harness clone now checks out the exp_36 folder); `test_exp36_chain.py` reuses that clone (-97 = its own clone fixture removed) | 0 | +146 / -97 | +572 |
| `7c3c973a` | 7 GREEN | generator RPL-S42 / OFF-S42 (386 rows); `EXP36_NAMES` validator ahead of the exp_35 name/block check; runner exp_25-only jq sets exp_36 rows apart (mutation-checked) | +29 / -13 | +2 / -1 | +154 / -4 |
| `bd493823` | 8 RED | the four replays through one registry (YAW / P1 / CANON at d6ca307a, B-F at H35), canon sidecar, backend proofs, multi-replay plans | 0 | +124 / -6 | +655 |
| `3f62b3c7` | 8 GREEN | `EXP36_REPLAYS`, generalized reference / check, CLI `--exp36-replay / --exp36-ref-dir`; all four REAL references verify | +102 / -52 | +4 / -2 | +177 |
| `3cf6ed70` | 9 RED | chain modes `offorbit` (chain I) / `A` (chain II), lanes 0,1, pinned CANON checkpoint manifest | 0 | +81 / -13 | +631 |
| `461c58e0` | 9 GREEN | **H36_I** (the launch commit of chains I and II): `yaw_c4_spread_chain.sh {offorbit,A}` | +54 / -31 | +3 / -3 | +157 |
| `d2076042` | 10 RED | `--exp36-set A / OFF`, YAW-verdict gate, base-dir-relative plan | 0 | +49 / -6 | +606 |
| `a198d46d` | 10 GREEN | publication sets (48 / 12 files, one manifest per set); cross-version verdict guard vs 461c58e0 | +29 / -19 | +14 | +100 |
| `071a52fa` | records | ledger, notebook entry, command file, H35 baseline suites, run-worktree links, the chain I / II DRY_RUN transcripts, publish gate smoke | 0 | 0 | +3155 |
| `8f2c290b` | docs | two docstrings brought up to date (no behaviour change) | 0 (docstrings) | 0 | 0 |
| `2881a3ad` | 11 RED | Codex code review I HIGH 1 follow-ups: parsed CLI OFF (12 files, own manifest), per-set manifest paths, publisher provenance | 0 | +23 | +24 |
| `0d99eeaf` | 11 GREEN | publication manifest records `publisher` = {repo, HEAD} beside `pin` / `campaign_digest` | +3 / -1 | 0 | +78 |
| `5ee1d6fe` | records | the suites at H36_I / the dev head, the comparator CLI smoke, notebook, ledger | 0 | 0 | data only |
| `7f5af103` | 12 RED | Codex code review II HIGH 3 + NIT 1-3: `publisher_closure` / `--publisher-pin`, own cells only, per-set default manifests (no override), the cross-version relative-plan guard, help A 48 / OFF 12; shared helpers adapted | 0 | +143 / -24 | +784 |
| `a5f0f898` | 12 GREEN a | `publisher_closure()` + required `--publisher-pin`; `--publish-manifest-out` removed; help (two HIGH 2 tests stay RED until 13) | +44 / -15 | 0 | +127 |
| `2f6cc928` | 13 RED | review II HIGH 1 + HIGH 2: global lock, concurrent same-set / A-OFF / path swap, txid temps, exclusive + authenticated manifest, resume refusals | 0 | +168 | +363 |
| `8628dbc2` | 13 GREEN 1/2 | the publication helpers only (lock, once-opened destination, exclusive no-follow writes, identities, `_Manifest`, `_authenticate`); not wired (split for the < 200 rule) | +99 | 0 | +4 |
| `3d2246d0` | 13 GREEN 2/2 | `exp36_publish` wired to them (the last code commit of the fix round) | +104 / -89 | 0 | +140 |
| `b1f6194b` | records | the review-II fix-round suites (760 passed) + gate smoke, ledger, notebook, command file | 0 | 0 | data only |
| `62a0841b` | 14 RED | the Planner's addendum (dumps ON): dump preflight, the replay with its dump + CTL-style dump checks, sources with dumps, the chain's runner args (273 test lines: every replay / publication world now dumps) | 0 | +196 / -77 | +2346 |
| `d92aef10` | 14 GREEN | **H36_I2** = the new launch commit of chains I and II: the chain without `--no-dump-blocks` + `--exp36-dump-preflight` before each runner step; the references' dump pins; the replay verified WITH its dump + CTL-style checks; sources verified with dumps | +78 / -15 | +17 / -8 | +267 |

Every code commit is < 200 changed code lines; one test commit (`65f9fcd4`, 243 test lines) exceeds 200 because it
removes a duplicated clone fixture (-97) while adding the A1 tests.
