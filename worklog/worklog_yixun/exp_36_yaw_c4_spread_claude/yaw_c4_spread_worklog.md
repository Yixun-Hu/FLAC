# exp_36 yaw_c4_spread — notebook

## 2026-11-06T14:52:04-05:00 — Planner: the one Codex plan review = APPROVE-WITH-CHANGES → the Coder round starts
- Review: `yaw_c4_spread_codex_plan_review.md` (verbatim). Binding (in the brief `yaw_c4_spread_coder_brief_r1.md`, which names the review as overriding the plan): B1 `uses_no_flash()` separate from `is_exp35()` (`EXP35_BLOCKS` unchanged) + an exp_36 name validator; B2 explicit generator name/tag control, block `YAW-C4S` = 25 rows (orders 346–370, priority 18, replay first), the H35 render (`build_manifest(exp36=False)`, sha256 `440eb1be…`) and the exp_25-only render (`1073ef64…`) preserved by regression tests; B3 a cross-worktree replay verification (reference at the exp-25 worktree, pin `d6ca307a`, dump expected; replay at the exp_36 worktree, pin H36, dump-free, campaign digest; the three archived raw digests verified first; a SHA-bound verdict); B4 one fail-closed chain replay → comparator → the 24 (exit 0 required), publish re-checks or consumes the verdict; H1 `YAW-C4S` out of default selection; H2 no further key excluded from equality; H4 hardened publication (pinned plain destination, exp-25 HEAD, 48-path preflight, atomic no-replace, never the replay or receipts, recovery); H5 A on ONE fixed card (GPU 1), B on GPU 0, `ALLOW_SHARED_GPU` unset, RELEASE only by the A/B coordinator (the Planner) after both; H6 the exp_36 records imported into the leaf branch, the runner's log locations handled; H7 the paper script needs `YAW_SEEDS = 42–46` AND its seed-42 labels changed (Yixun's / the desktop session's paper-repo edit).

## 2026-11-06T16:20:00-05:00 — Coder r1 (Opus 5.5): the exp_36 code round + amendment A1, TDD cycles 1-10
- **Goal** — implement the brief r1 (plan r0 as overridden by the Codex plan review B1-B4 / H1-H7) and the Planner's
  amendment A1 (Yixun 15:30 / 15:32: the seed-42 off-orbit run FIRST), CPU only, no GPU work.
- **Change** — `depth_canon_make_cells.py` (+ `depth_canon_cells.json`, 386 rows): YAW-C4S (25: the replay first, the
  24 aliases), RPL-S42 (3 replays P1 / B-F / CANON), OFF-S42 (6 exp35_*_th315 + 6 exp25_YAW_C4_*_s42_th{135,225,315});
  explicit names only for exp_36 rows; `build_manifest(exp36=False)` = the H35 render, `exp35=False` = the exp_25 render.
  `depth_canon_collect.py`: `uses_no_flash` (EXP35 + EXP36 blocks) for no_flash / receipt 2 / backend log /
  `build_receipt` / `--count-no-flash` (`is_exp35`, `EXP35_BLOCKS` unchanged); `EXP36_NAMES` validator (ahead of the
  exp_35 name/block check); `DEFAULT_EXCLUDED_BLOCKS` = E2-FAIL + the exp_36 blocks (`EXPLICIT_BLOCKS` stays E2-FAIL);
  `EXP36_REPLAYS` registry + `exp36_reference` / `exp36_replay_check` / `--exp36-replay-check`; `exp36_publish` /
  `--exp36-publish --exp36-set A|OFF`. `depth_canon_eval_run.sh`: the chain script in the relevant tree of every no-flash
  selection; the exp_25-only model-config jq sets exp_36 rows apart. New `yaw_c4_spread_chain.sh {offorbit|A}`.
- **Version Control** — branch `exp-36-yaw-c4-spread` from H35 `5b065801`; commits in `commits_yaw_c4_spread.md`;
  H36_I = `461c58e0a70747d163e554ebb123a5869d3a38c2` (the launch code of chains I and II); publication code at `a198d46d`.
- **Command / Validation** — every cycle RED then GREEN, logs `yaw_c4_spread_<ts>_C<n>_*_{red,green}.log`; baseline
  suites at H35 (`*_BASELINE_suites_at_H35.log`): 1513 passed / 1 skipped (CUDA) / 0 failed; the full suites at H36_I
  (`*_SUITES_at_461c58e0_in_the_run_worktree.log`); the DRY_RUNs `*_DRYRUN_chain_I_offorbit_at_461c58e0.log` (28 cells)
  and `*_DRYRUN_chain_II_A_at_461c58e0.log` (25 cells), each with a python argv-vs-template verification appended; the
  publish gate smoke `*_publish_gate_smoke.log` (real worktrees, refused at the verdict, nothing written).
- **Result** — passed. All four REAL archived references verify read-only against the amendment's digests (YAW / P1 /
  CANON incl. the canon sidecar at d6ca307a in the exp-25 worktree; B-F at H35 in the exp-35 run worktree, its
  no-flash launcher log); pinned log digests YAW 8cbf6e24, P1 a9e2dff3, CANON a201c970 (flash_attn-not-installed
  fallback pair), B-F af938eb2 (launcher sentinel). Both DRY_RUNs exit 0, one evaluator tree digest `49daa868…`, every
  cell through the no-flash launcher, no dump flag, argv = template except `--seed` / `--rotate-deg` / `--eval-name`.
- **Analysis** — one real defect found and fixed (test side): cycle 5 put the chain into the runner's relevant tree;
  the shared SPARSE harness clone of the exp_35 runner tests did not check the exp_36 folder out, so `git ls-files`
  named a file `sha256sum` could not read (TREE DIGEST FAILED for every exp_35 no-flash harness run). Production
  checkouts are never sparse; the clone now includes the folder. One latent regression prevented (cycle 7): the
  RPL-S42 B-F replay would have pulled FLAC_AR_BF.json into an exp_25-only relevant tree (exp_35 C3a test,
  mutation-checked). Deviation (A1 §4 over Codex H5): both chains use `--lanes 0,1`; ALLOW_SHARED_GPU stays 0.
  Records location (Codex H6): the runner keeps its logs / plans in the exp_25 folder, the chain logs and verdicts are
  in the exp_36 folder; both kept, the chain log names the runner's files.
- **Next** — the Planner launches chain I from the DETACHED run worktree `~/codespace/exp-36-run` (HEAD H36_I; its own
  asset links; created so that branch commits never move a running campaign's HEAD), then chain II there; then the
  two publications from this worktree with `--base-dir ~/codespace/exp-36-run --pin 461c58e0…` (commands in
  `yaw_c4_spread_command.md`); a Codex code review of this round before the launch is the Planner's call.

## 2026-11-06T16:25:00-05:00 — Coder r1: Codex code review I HIGH 1 (publication sets) closed in code; verification at the dev head
- **Goal** — close review I's HIGH 1 before any publication (Chain I unaffected: it never publishes).
- **Change** — cycle 10 (`d2076042` RED / `a198d46d` GREEN): `--exp36-publish --exp36-set A|OFF` (`EXP36_PUBLISH_SETS`:
  the 24 YAW-C4S cells / the 6 YAW s42 off-orbit cells; 48 / 12 destination paths), one default manifest per set
  (`yaw_c4_spread_publication_manifest_{A,OFF}.json`), the YAW replay's verdict only, the replay plan resolved against
  `--base-dir` (the run worktree's cells published from this checkout, pin 461c58e0, its digest kept as the artifacts'
  provenance); a cross-version guard: the collector at 461c58e0 and the current one compute the same verdict dict (the
  publication re-computes and compares it). Cycle 11: the parsed-CLI OFF test (12 files, own manifest), the per-set
  manifest paths, and `publisher` = {repo, HEAD} recorded beside `pin` / `campaign_digest`.
- **Command / Validation** — the comparator CLI smoke at H36_I in the run worktree on the chain-I DRY_RUN plan
  (`*_comparator_cli_smoke_at_461c58e0.log`): all four REAL references pass, each replay absent -> REFUSED rc 2 by
  design, no file written; the publish gate smoke (real worktrees) refused at the verdict, nothing written.
- **Next** — the Planner's chain I launch waits on PAUSED (orbitrir-haa holding); chain II and both publications follow.

## 2026-11-06T16:50:00-05:00 — Coder r1: verification complete; branch pushed
- **Result** — passed. Full depth_canon + exp_35 + exp_36 suites at H36_I `461c58e0` in the run worktree
  (`*_SUITES_at_461c58e0_in_the_run_worktree.log`): 1645 passed, 1 skipped (needs a CUDA device), 0 failed (39 min).
  At the dev head `8f2c290b`: the exp_36 + collector + runner suites 635 passed / 0 failed
  (`*_SUITES_exp36_and_collector_at_8f2c290b_dev_head.log`), the remaining kit suites 242 passed / 0 failed
  (`*_SUITES_rest_at_8f2c290b_dev_head.log`); cycle 11 (`2881a3ad` / `0d99eeaf`) GREEN 65 passed. H35 baseline for
  comparison: 1513 passed / 1 skipped / 0 failed. The comparator CLI smoke at H36_I (all four REAL references pass;
  absent replays refused rc 2) in `*_comparator_cli_smoke_at_461c58e0.log`. Temporary baseline worktree
  `~/codespace/.exp36-baseline-H35` removed (its three symlinks first; targets intact).

## 2026-11-06T17:25:00-05:00 — Coder r1, fix round after Codex code review II (REQUEST-CHANGES, no BLOCKING; round 2 of 3)
- **Goal** — fix every finding of review II that is the Coder's (HIGH 1-3, NIT 1-3; HIGH 4 / NIT 4 are the Planner's
  records), each covered by tests; publication code only -- H36_I `461c58e0` and the run worktree stay untouched.
- **Change** — HIGH 1: ONE global nonblocking publication lock = flock on the exp-25 worktree root directory (no lock
  file; every publisher of either set from any checkout contends on that inode, which a swap of the YAW path does not
  replace), held from before the first check through the final manifest + directory fsync; the destination opened once
  O_DIRECTORY | O_NOFOLLOW (inode == path), every temp creation / link / hash / unlink relative to that fd; a transaction
  id per publication names its temps; a path swap ends the owner as `partial` (it never writes through the new path).
  HIGH 2: the manifest created O_CREAT | O_EXCL | O_NOFOLLOW (file + directory fsync), rewritten after every file through an
  exclusive no-follow temp that replaces ONLY the still-authenticated manifest (inode + bytes); a resume requires a
  regular non-symlink manifest of exactly this publication (check, set, pin, digest, exp-25 HEAD, base dir, destination,
  verdict path + sha256, publisher, the complete entry set incl. source hashes and txid-named temps, status in_progress |
  partial) with no writer temp beside it, and removes only the temps it names; a complete publication is final; the
  CLI writes only the set's fixed default manifest (`--publish-manifest-out` removed). HIGH 3: a required
  `--publisher-pin` (full SHA) checked by `publisher_closure` -- HEAD == pin and the closure (eval_FLAC.py, src/ but its
  tests, the collector, the cells manifest) committed and clean; the manifest records `publisher` = {repo, head,
  collector path + sha256, closure digest} beside the artifacts' `pin` (461c58e0) + `campaign_digest`; the publication
  reads only the collector's own cells manifest. NITs: the parsed CLI's per-set default routing tested (A 48 / OFF 12,
  two files), the cross-version guard now = old collector + relative plan in the run checkout vs this collector + the
  same string from another cwd (parsed-dict equality), help + test header name both sets.
- **Version Control** — `7f5af103` RED, `a5f0f898` GREEN a, `2f6cc928` RED, `8628dbc2` + `3d2246d0` GREEN (split for the
  < 200-line rule: 99 / 193 code lines). Since 461c58e0 the runner, the chain, the generator, the cells manifest, the
  overlay, eval_FLAC.py and src/ are byte-identical; the collector changed only in the publication section, its CLI and
  (cycle 10) the base-dir-relative plan read, verdict-neutral (cross-version guard).
- **Result** — passed: the publication suites 96 passed (`*_C13_lock_manifest_green.log`); the read-only gate smoke
  with the real worktrees (`*_publish_gate_smoke_after_review_II.log`): lock, publisher closure, HEADs and destination
  pass, refused at the verdict, a wrong / missing publisher pin refused, nothing written. The full post-fix suites:
  `*_SUITES_exp36_collector_runner_at_3d2246d0.log`.
- **Next** — Codex code review III (the last) of `5ee1d6fe..` before either publication.

## 2026-11-06T18:20:00-05:00 — Coder r1: the Planner's addendum (every exp_36 cell dumps) -> the new launch commit H36_I2
- **Goal** — Yixun's announcement 08 + A1 ("exactly the flags of its existing cells"; the templates all dumped): drop the
  plan's "dumps off" for every chain-I / chain-II cell, in the same revision as the review-II fixes.
- **Change** (cycle 14: `62a0841b` RED / `d92aef10` GREEN) -- the chain: no `--no-dump-blocks` (each row's own
  `dump: true`, the manifest's NAS dump root) and `--exp36-dump-preflight` before each runner step (refuses a target dump
  directory or `.staging-` sibling without this checkout's receipt; a resumed chain's own dumps are left to the runner's
  reviewed predicate). The comparator: replay plans must plan the replays WITH dumps; the replay verified with its dump
  (`dump_check full`); the exp_35 CTL-style dump checks (waveform sha256 + COMPLETE marker vs the reference's pinned
  waveform digest, n / dtype / shape, input_hash vs the reference dump manifest); `EXP36_REPLAYS` pins each reference
  receipt's dump evidence (YAW 6a702f00 / 1485ef84, P1 99de9897 / 34dfa66d, B-F 975cb721 / d9f46deb, CANON ca864c12 /
  441330a0). The publication: sources verified with their dumps (receipt evidence, offline); dumps are never copied.
- **Between H36_I `461c58e0` and H36_I2 `d92aef10`** -- changed: `yaw_c4_spread_chain.sh` (dumps on + preflight),
  `depth_canon_collect.py` (cycle 10 publication sets + the base-dir-relative plan read; cycles 12-13 publication
  hardening; cycle 14 the comparator's dump checks, `exp36_dump_preflight`, the publication's dump-ful sources), tests.
  Unchanged byte for byte: `depth_canon_eval_run.sh`, `depth_canon_make_cells.py`, `depth_canon_cells.json` (every
  exp_36 row already says `dump: true`), `overlays/no_flash/*`, `eval_FLAC.py`, `src/`.
- **Result** — the run worktree moved to H36_I2 (`*_run_worktree_moved_to_H36_I2.log`; no process there); both DRY_RUNs
  exit 0 (28 / 25 cells, every argv = its template's + `--ckpt-path` + `--dump-scored-waveforms <NAS root>`, all through
  the no-flash launcher, the dump preflights OK against the real NAS, no quarantine, one tree digest `b4795f8b...`);
  tests GREEN (154 passed / 1 skipped until the launch commit was recorded); the full suites at H36_I2:
  `*_SUITES_at_H36_I2_d92aef10_in_the_run_worktree.log`. Note: the 12 H35 OFF rows' dump names are exp_35's; if exp_35
  phase (ii) (superseded) were ever run into the same dump root, its runner would quarantine (rename) these dumps.

## 2026-11-06T18:50:52-05:00 — Coder r1: Codex code review III (the last; cap) step 1 -- the replacement launch commit H36_I3
- **Goal** — the Planner's priority interrupt: review III HIGH 1 + NIT 1 + NIT 2 in a replacement launch commit before
  chain I can launch; every fix is "fixed after review III, not re-reviewed (cap)".
- **Change** (cycle 15: `9a8440dc` RED, committed 18:32:37 EST / `6be49f46` GREEN, committed 18:49:40 EST) -- HIGH 1: the
  dump preflight takes the launch pin (`--pin`; the chain passes `EXPECT_HEAD`) and treats an existing dump directory or
  `.staging-` sibling as this checkout's only when the cell's receipt AUTHENTICATES: read through one no-follow,
  non-blocking descriptor, a regular file, parseable JSON, `cell` + `stem` of the selected cell, `expect_head` = the
  launch pin, `ckpt_link` under the farm root, `dump_root` = the dump root, `dump: true`; such an owned dump, complete
  or damaged, stays the runner's reviewed predicate (SKIP if verified, else QUARANTINE + rerun); anything else collides.
  NIT 1: the chain prints only a pointer to `yaw_c4_spread_command.md` of the publisher checkout (+ the pin, base dir
  and verdict the command needs); a printed publication command lacking `--publisher-pin`, `--base-dir` or the
  publisher checkout fails the chain test. NIT 2: the comparator / test descriptions say every replay dumps, the
  comparison checks waveforms + markers, publication verifies dump evidence offline and never copies a dump.
- **Result** — GREEN: 167 passed (the exp_36 modules), then the preflight + chain modules 65 + 23 passed after the
  single-descriptor read; 11 mutation checks of HIGH 1, all caught (one survived first -- the test's alarm raised an
  OSError subclass the open swallowed; fixed in the test, re-run: all caught). H36_I3 recorded in
  `yaw_c4_spread_launch_commit.txt` (this commit).
- **Next** (not done at this entry's commit) — move `~/codespace/exp-36-run` to H36_I3 after the /proc + lsof check
  (my own H36_I2 suite still runs there, ETA ≈ 18:56 EST), repeat both DRY_RUNs, send the Planner the chain commands;
  then HIGH 3 + NIT 3 (step 2); then the exp_37 B3 leaf on top.

## 2026-11-06T19:00:29-05:00 — Coder r1: review III step 1 done -- the run worktree at H36_I3, both DRY_RUNs repeated
- **Result** — my H36_I2 full suite in the run worktree finished 18:54:44 EST: 1704 passed, 2 skipped (no CUDA; the
  cross-version guard, skipped because the launch commit was recorded only after that suite started), rc 0
  (`*_SUITES_at_H36_I2_d92aef10_in_the_run_worktree.log`; superseded by H36_I3). At 18:55:08 EST no process had its cwd
  in `~/codespace/exp-36-run` (/proc/*/cwd scan and lsof +D: none); it was detached from H36_I2 to H36_I3 `6be49f46`,
  0 tracked changes (`*_run_worktree_moved_to_H36_I3.log`). DRY_RUNs at H36_I3: chain I offorbit 18:55:47-18:58:02 EST
  exit 0, 28 cells; chain II A 18:58:13-18:59:41 EST exit 0, 25 cells; every argv = its row's eval_args +
  `--ckpt-path <farm link>` + `--dump-scored-waveforms <NAS root>`, all through the no-flash launcher; each dump
  preflight called with `--pin 6be49f46...` and OK against the real NAS; no quarantine; one tree digest `f4b7056e...`.
- **Between H36_I2 `d92aef10` and H36_I3 `6be49f46`** -- changed: `depth_canon_collect.py` (`_owned`, the preflight's pin
  + authenticated-receipt rule, its CLI help, the comparator docstring), `yaw_c4_spread_chain.sh` (`--pin` to the
  preflight, the publication pointer, a header comment), tests, records. Byte-identical: the runner, the generator, the
  cells manifest, `overlays/no_flash/*`, `eval_FLAC.py`, `src/` (but `src/tests`).
- **Next** — send the Planner the chain commands (`yaw_c4_spread_command.md`); then review III HIGH 3 + NIT 3.

## 2026-11-06T19:10:42-05:00 — Coder r1: Codex code review III step 2 -- HIGH 3 (publication-manifest cleanup) and NIT 3 (chronology)
- **Change** (cycle 16: `ecd4ee64` RED, committed 19:07:17 EST / `20200cb4` GREEN, committed 19:09:15 EST; fixed after
  review III, not re-reviewed (cap)) -- `_write_new` writes with `os.write` + fsync + close and, when any of them
  fails, removes the file it created (only while the name is still that inode: `_unlink_created`); `_Manifest.write`
  removes its own writer temp in a `finally` unless `os.replace` consumed it, and takes the replaced manifest's identity
  before the directory fsync; the CLI's crash message: no manifest = nothing published, re-run the same command; a
  manifest = `--exp36-publish-resume`; the recovery text: a writer temp left by a hard kill inside an update is never
  removed automatically (resume refuses until a reviewed manual step removes it). Foreign writer / destination temps are
  still refused. The chain, the runner, the generator, the cells manifest, the overlays, `eval_FLAC.py` and `src/` are
  unchanged since H36_I3: the launch commit stays H36_I3 `6be49f46`; the publication code is `20200cb4`.
- **Result** — GREEN: 84 passed (the five publication modules, the cross-version guard ENABLED: the H36_I3 collector's
  verdict reproduced by this one); 5 mutation checks of HIGH 3, all caught (`*_C16_review_III_failures_green.log`).
- **Chronology correction** (Codex code review III NIT 3; appended, nothing above rewritten). Three entries above carry a
  header time LATER than the commit that holds them (completion estimates written before committing):
  - "2026-11-06T16:20:00" is in `071a52fa`, committed 16:13:56 EST. At that commit the full suites at H36_I (started
    16:07:29) were still running; their result (1645 passed, 1 skipped; finished 16:46:48) was first recorded in the
    "16:50:00" entry. The "Result — passed" of the 16:20 entry covers the checks it lists that had finished, not that suite.
  - "2026-11-06T16:50:00" is in `5ee1d6fe`, committed 16:48:13 EST; its "branch pushed" happened after that commit (the
    push updated origin at 16:48:22 EST).
  - "2026-11-06T18:20:00" is in `d373c18e`, committed 18:16:37 EST (the entry the review names). At that commit the
    full suites at H36_I2 (started 18:15:54) were running; their result (1704 passed, 2 skipped; finished 18:54:44) is
    in the 2026-11-06T19:00:29 entry.
  From the 2026-11-06T18:50:52 entry on, each header is the clock time at writing, at or before its commit.
- **Next** (not done at this entry's commit) — the full suites at this checkout's head; the exp_37 B3 leaf rebased onto
  this head.
