# exp_35 orbitrir_yaw_evidence — Coder log (Claude Opus 5.5, max effort)

Worktree `~/codespace/exp-35-orbitrir-yaw`, branch `exp-35-orbitrir-yaw`, cut from `localization-exp` @ `682df5db`
(plan r2). Brief: `scratchpad/r3a/exp35_coder_brief_launch.md`. Plan updates received mid-task from the Planner:
r3 (§12, main checkout `2cf63019`) and r4 (§13, `e2166326`); this worktree's copy of the plan stays at r2 —
the binding text is read from the main checkout with `git show`. CPU only throughout (`CUDA_VISIBLE_DEVICES=""`,
`PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`); no environment change; no process of anyone else touched.
Timestamps from `date`. Logs: `orbitrir_yaw_evidence_<ts>_<what>.log` in this folder.

## 2026-11-05T11:52:04-05:00 — baseline
- The kit suites before any change: `test_depth_canon_eval_run.py`, `test_depth_canon_collect.py`,
  `test_depth_canon_cond_probe.py`, `test_gen_model_comparison_depth_canon.py` → **364 passed** (9 min 44 s;
  `orbitrir_yaw_evidence_2026-11-05_11-52-04_baseline_kit_suites.log`).

## 2026-11-05T12:09:18-05:00 — round 1a: the no-flash overlay + launcher (`fb496c3e`)
- **Goal** — plan r2 §11 B1 steps (1)–(7), H5, and r3 §12 H4 (a)/(b)/(c): a single-file overlay
  `overlays/no_flash/flash_attn.py` (the §3b four lines verbatim, sha256 `0d0e52ca…c023` — unchanged by the rename
  from `flash_attn/__init__.py`, so the plan's verified value holds), the same-interpreter launcher
  `overlays/no_flash/exp35_no_flash_launch.py` (sha256 `980051ac…151d`), `pyproject.toml`
  `exclude = ["overlays*"]`.
- **RED** `orbitrir_yaw_evidence_2026-11-05_12-06-24_r1a_launcher_red.log` (30 failed: no launcher) →
  **GREEN** `…_12-09-18_r1a_launcher_green.log` (33 passed; the two intermediate green logs record the two
  test fixes below).
- **Decisions / ambiguities resolved**
  - "REFUSES … if a `flash_attn.py` / `flash_attn/` exists at the repo root or anywhere earlier on the path it
    builds" (§11 B1 (2)): the overlay is first on the built path, so "earlier" is read as *ahead of site-packages*
    (the installed package's location): any `flash_attn` / `flash_attn.*` entry at the repo root or in any path
    entry before the first site-packages directory is refused; the overlay directory itself must hold EXACTLY the
    two committed files (so a `__pycache__`, a `*.pyc`, a `flash_attn/` package or an extension module there is
    refused before any import), and nothing under `overlays/` may be a package marker or bytecode.
  - "drops any inherited PYTHONPATH entries" + §12 H4 (c) "the runner ALSO exports PYTHONPATH=<repo>/overlays/
    no_flash": the launcher removes every inherited entry from `sys.path` and sets `PYTHONPATH=<overlay dir>` for
    its children (so a fresh Python child also meets the overlay first).
  - Step (4) is checked twice: the spec (origin = the overlay file, a source module, its bytes = the pin) and an
    explicit import that must raise exactly the overlay's ImportError and leave nothing in `sys.modules`; then
    step (5) imports `src.models.transformer`, which re-runs the overlay and prints the pair the per-cell check
    reads. Step (5) also checks `src` itself is `<repo>/src`: the flac env holds a stale installed `src` package in
    site-packages (found by a test; the repo root ahead of site-packages makes the checkout's win, the check refuses
    otherwise).
  - Refusal exit code 86 (distinct from eval_FLAC's argparse 2); `--preflight` runs (1)–(6) and stops.
  - No test imports the launcher or the overlay in-process (it would write bytecode next to them, which the
    launcher then refuses): every case is a subprocess; most use a throwaway repo with a stand-in transformer
    that carries the real file's fallback block verbatim.
- **Process note** — the first commit picked up a stale staged copy of the test file; amended before anything
  else (local, unpushed) → `fb496c3e`.

## 2026-11-05T12:32:25-05:00 — round 2: the generator (`1ba565bc`)
- **Goal** — the 100 exp_35 rows (plan r2 §1; orders 230–329; priorities 10–16 for CTL, BF-pin, BF-dup,
  OFF-P1, OFF-BF, OFF-CANON, CANON-45) in the exact exp_25 26-field row schema; the BF arm; the exp_25-only render
  byte for byte.
- **RED** `…_12-17-20_r2_generator_red.log` (the implementation reverted: 7 failed + 4 errors) → **GREEN**
  `…_12-18-59_r2_generator_green.log` (44 passed) → kit suites `…_12-20-35_r2_suites_green.log` (**409 passed**).
- **Guard** — `render(build_manifest(exp35=False))` sha256 `1073ef647434…0575`, equal to
  `git show d6ca307a:…/depth_canon_cells.json`; the full 330-row file sha256 `a1fbebd5be5f…0e0e0`.
- **Decisions / ambiguities resolved**
  - Names: CTL `exp35_P1_T_S40000_K{k}_s42` and B-F θ=0 `exp35_BF_T_S40000_K{k}_s{s}` exactly as the plan writes
    them (§3c, §4); `exp35_BF_R_…`, `exp35_BF_C4_…_th{d}`, `exp35_BF_T_…_s42_dup`; every off-orbit cell (OFF-* and
    CANON-45) carries the tag `OFF` + `_th{d}`.
  - The B-F argv replaces the two leading `--cond-method vanilla` tokens of COMMON_FLAGS with
    `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64` (§1 B-F flags);
    the full manifest's header gains `arm_cond_flags` and the BF arm; the exp_25 header stays unchanged.
  - Manifest order vs §7's chunk order: within plan-named blocks the dup block (priority 12) follows BF-pin's
    C4+45 rows, while §7 lists "T s43–46, R, dup, C4+45". The manifest keeps §1's block layout (exp_25
    conventions, K-major); the frozen 33-chunk order of §7 belongs to the supervisor (round 6a).
  - In the same commit the collector's `--collect` (the exp_25 mode) selects the exp_25 blocks by default and
    refuses an exp_35 block (§3 P6): without it the exp_25 full-collection tests would demand the 100 new rows.

## 2026-11-05T12:34:33-05:00 — round 3: the collector protocol (`da88eb38`)
- **Goal** — one per-arm protocol table (§3 P2): `ARM_PROTOCOL` (B-F `fa_invariant`, `[0.0, 90.0, 180.0, 270.0]`,
  `batched`, cap 64, bf16; vanilla by default) driving `record_reasons` (the exp_25 expectation dict keeps its
  key order and values) and `cell_paths` (the `_fa_invariant_a4` stem); the shared fixture writes a B-F cell as
  eval_FLAC records an fa_invariant evaluation.
- **RED** `…_12-32-36_r3_protocol_red.log` (13 failed) → **GREEN** `…_12-32-52_r3_protocol_green.log`
  (22 passed) → kit suites at `da88eb38` `…_12-35-17_r3_suites_green_at_da88eb38.log` (**432 passed**; run on a
  scratch clone of the commit, the file was empty while that run was in flight when Codex C1 read it).
- **Disclosure (Codex C1 NIT 2)** — the campaign-consistency check (an `exp35_` name in a non-exp_35 block is
  refused) went into the round-3 patch with the protocol; its test was written afterwards, so its evidence is
  GREEN (`…_12-33-28_r3_campaign_check_green.log`) followed by a mutation RED with the check deleted
  (`…_12-33-48_r3_campaign_check_mutation_red.log`), not a chronological RED → GREEN.

## 2026-11-05T12:36:38-05:00 — round 1b: the backend evidence in the collector (`65c60dd8`)
- **Goal** — §3b / §11 B1, the collector half: `NO_FLASH` (overlay + launcher sha256, the log lines);
  `backend_log_reasons` (exactly one back-to-back pair of the overlay ImportError line and transformer.py's
  `flash_attn not installed, disabling Flash Attention`; exactly one launcher sentinel after it naming `sdpa`, both
  sha256 and the target; no `No module named 'flash_attn'`); `--check-backend-log`; receipt schema 2 for exp_35
  cells only (`attention_backend`, `no_flash_overlay_sha256`, `no_flash_launcher_sha256` appended);
  `build_receipt` refuses missing / wrong backend fields and any on an exp_25 cell; the log re-read at every
  `verify_cell`; plan entries carry `no_flash`.
- **RED** `…_12-35-26_r1b_backend_receipt_red.log` (32 failed) → **GREEN** `…_12-35-56_r1b_backend_receipt_green.log`
  (55 passed with round 3's file) → kit suites at `65c60dd8` `…_12-36-49_r1b_suites_green_at_65c60dd8.log`
  (**464 passed**).
- **Decisions** — one predicate, used twice (the runner's per-cell check and `verify_cell`'s re-check); the
  log check binds the launcher target (default `eval_FLAC.py`; the runner passes `$EVAL_PROG`, a stub in the
  harness); the predicate requires exactly ONE transformer pair (eval_FLAC's DataLoader workers fork, so a
  second pair would mean a second import route).
- **Order** — rounds 1b and 1c were committed after rounds 2 and 3 because the runner's end-to-end tests need
  the exp_35 rows and receipt v2.

## 2026-11-05T12:54:31-05:00 — round 1c: the identity helper and the runner (`9d4e00b6`, `0e4efbc0`)
- **Goal** — §11 B1 runner half + r3 §12 B1.1 / H4 (c) (forwarded by the Planner): exp_35 cells spawned through
  the launcher with `PYTHONPATH=<repo>/overlays/no_flash` only and `PYTHONDONTWRITEBYTECODE=1`; RELEVANT += the
  overlay dir + the identity helper; both sha256 runner constants checked at start; the CPU pre-flight
  (`--preflight` + `--check-backend-log --backend-target=--preflight`) before any spawn, DRY_RUN too; after rc 0
  the backend check or rc 5 → the campaign hard-stops (exit 2, no receipt); receipt flags for schema 2; the
  identity tuple (`EXP35_APPROVED_COTENANTS`) enforced in `lane_state` at the selection AND the final pre-spawn
  check (refusal exit 2); exp_25 cells keep their spawn line byte for byte.
- **Identity helper** — drafted before its tests, then moved aside for an honest RED
  (`…_12-27-38_r1c_identity_red.log`, collection error) → GREEN `…_12-27-40_r1c_identity_green.log` (23 passed).
- **Runner** — RED with the old runner `…_12-39-45_r1c_runner_red.log` (14 failed, the exp_25 case passing) →
  GREEN `…_12-46-15_r1c_runner_green.log` (15 passed).
- **Decisions / ambiguities resolved**
  - The approved tuple must EQUAL the observed foreign set, so a lane whose trainer exited (empty set) is refused
    too (§12 B1.1 "missing"); a tuple without `ALLOW_SHARED_GPU=1` and `ALLOW_SHARED_GPU=1` for an exp_35 cell
    without a tuple are both refused at start.
  - A backend-check failure is rc 5 (a new code inside the runner) mapped onto the existing ABORT path, so the
    runner's exit codes stay 0 / 1 / 2 / 4.
  - The shared harness stub now records how it was launched (PYTHONPATH, sys.path[:2], bytecode flag, whether
    the SDPA transformer was already imported); the clone fixture adds `overlays` + the exp_35 folder.

## 2026-11-05T12:57:41-05:00 — Codex code C1 (REQUEST-CHANGES) fix (`f5a980bf`)
- **HIGH 1** — the launcher compared realpath with realpath, so a symlinked `src/models/transformer.py` (or
  `src/__init__.py`, or a symlinked `src/` / `src/models/`) leading out of the checkout passed. Now every file on
  the transformer's import chain (`src`, `src.models`, `src.models.transformer`) must have
  `realpath(__file__)` equal to its LEXICAL path under the canonical repo. RED `…_12-55-18_C1fix_launcher_red.log`
  (7 failed: five restored symlink cases, the outside-`src` wording, the direct-invocation test) → GREEN
  `…_12-56-45_C1fix_launcher_green.log` (71 passed with the backend tests). New launcher sha256
  `bbd2847d3a50685420750d066195c8656a1509b4e905883886c0568d244043de`, re-pinned in the collector and the runner.
- **NIT 1** — the target fixture reports `__file__`, `abspath(__file__)`, cwd and pid; the target's pid equals the
  launcher process's pid; against a direct `python target.py …` from the same cwd the arguments, `__name__`,
  the resolved script path and the cwd agree. One difference is asserted and disclosed: a direct run on
  Python ≥ 3.9 makes `__file__` absolute where `runpy.run_path` keeps the path as given — every consumer
  (eval_FLAC `source_sha`, the probe's repo root) resolves it with abspath against the unchanged cwd.
- **NIT 2** — this log, `commits_orbitrir_yaw_evidence.md`, the round-3 suite log (non-empty now) and the
  disclosure above.

## 2026-11-05T13:06:39-05:00 — round 4a: the B-F composition probe, probe side (`c3c863bc`)
- **Goal** — plan r2 §5 "B-F probe": `depth_canon_cond_probe.py --arm BF`. A = FA(md) with no canonicalization,
  B = FA(rotate_d(md)) for d ∈ {128, 256, 384} (90 / 180 / 270 deg, keys d128 / d256 / d384), FA =
  `invariant_conditioning(conditioner, md, device, (0, 90, 180, 270), max_fwd_samples=64)`; the same four stages,
  protocols (fp32 / eval bf16) and determinism floor as the CANON probe; the first 64 unseen K=8 items, seed 42,
  batch 64.
- **Evidence** — RED `…_13-03-14_r4_probe_red.log` (12 failed) → `…_13-05-20_…_attempt1_test_reference_fixed_after.log`
  (1 failed: the test's stage-spy reference was computed outside autocast — a test bug) → GREEN
  `…_13-06-03_r4_probe_green.log` (113 passed with the existing cond-probe tests).
- **Decisions** — B's rotation is `rotate_scene_metadata` + `offsets_to_radians`, checked equal to eval_FLAC's fixed
  rotation (`math.radians(deg)` for 45 / 90 / 180 / 270); B-F takes no d64 and no random offset; `--arm BF` with
  `--allow-canon-mismatch` is a usage error; the dirty check covers `overlays/no_flash` for B-F; `img_w` must be 512;
  the B-F provenance records `cond_method`, `frame_avg_angles`, `frame_avg_max_fwd_samples` (no
  `random_offset_seed`); any deviation from the registered B-F run sets `registered = false`.

## 2026-11-05T13:39:18-05:00 — Codex code C2 (REQUEST-CHANGES) fixes (`38b809d6`, `e45ad2fe`)
- **HIGH 1 (`38b809d6`)** — `proc_identity` opens `/proc/<pid>` ONCE (`O_DIRECTORY`), reads stat / cwd / cmdline /
  fd 1 / fd 2 through that fd, and stats it before AND after; a different pid or start time → "changed during the
  sample". RED `…_13-07-52_C2fix_identity_red.log` (2 failed) → GREEN `…_13-08-10_C2fix_identity_green.log` (25).
- **HIGH 2 (`e45ad2fe`)** — the runner decides whether the selection holds exp_35 cells (`--count-no-flash`)
  BEFORE it builds RELEVANT; the overlay, launcher and identity helper are added and pinned only then; the exp_25
  plan entry keeps its shape (no `"no_flash": false`). RED `…_13-09-01_C2fix_exp25_independence_red.log` (9 failed)
  → GREEN `…_13-20-24_C2fix_exp25_independence_green.log` (9 passed); the tests remove / mutate the exp_35 files
  and compare the exp_25 plan and spawn byte for byte with the collector at `682df5db` (`_collector_at`).
- **BLOCKING 1 (`e45ad2fe`)** — the approved tuple is bound to its launch-entry sha256
  (`EXP35_APPROVED_COTENANTS_SHA256`, required) and its bytes are read once; the same digest is checked at every
  identity check; a final `lane_state` / identity check runs inside `run_cell` after the farm validation and the
  checkpoint sha256, immediately before the launcher (failure = hard stop, no receipt). Evidence:
  RED `…_13-22-11_C2fix_tuple_boundary_red.log` (5 failed, 1 passed — the `main_loop_recheck` case passed because
  its needle "hashes to" also matched an existing checkpoint message; needle tightened to "the launch entry pins")
  → RED `…_13-25-01_…_red2.log` (both replacement cases fail as they should) → GREEN
  `…_13-26-49_…_green.log` (46 passed, 1 failed: `test_the_tuple_must_carry_its_launch_entry_digest` passed
  `None` as an env value, which the harness cannot take) → `…_13-37-15_…` (still 1 failed: one needle for two
  different refusal messages) → `…_13-38-34_…` (passed with one needle per case). Production code unchanged across
  these three test-only reruns.
- **NIT 1** — the suite log at `f5a980bf` (`…_12-57-53_C1fix_suites_at_f5a980bf.log`) is non-empty now (508
  passed; it was still being written when C2 read it) and is committed here; the suite log at `e45ad2fe` is
  `…_13-39-30_C2fix_suites_at_e45ad2fe.log` (536 passed). Ledger / Coder-log updates stay in records commits.

## 2026-11-05T13:42:58-05:00 — round 4b: the B-F control, collector side (`1e252642`), and `a99eab15`
- **Goal** — §5 / §11 H3 / r3 §12 B3: `PROBE_CONTRACT_BF` mirrors the probe (offsets, keys, `fa_invariant`,
  angles, cap 64, checkpoint `5319feb4…` with 723,922,667 loaded bytes, config `3e677204…`, target); the probe JSON
  records the launch provenance (attention backend, overlay / launcher sha256, the H35 campaign digest, its log path
  and log sha256) through `finalize_bf_probe` — the supervisor's non-logged step after chunk 5: atomic JSON +
  regenerated `.sha256`, refusing a second pass, another arm, a wrong digest or a log without the backend evidence;
  `load_control_bf` returns the eval-protocol control p99 per offset + pooled and eps_cond (no verdict).
- **Evidence** — RED `…_13-41-00_r4b_contract_red.log` (44 failed) → `…_13-42-06_…_attempt1_one_test_moved_after.log`
  (1 failed: the "other arm" case belonged in the finalization-refusal test, moved) → GREEN
  `…_13-42-29_r4b_contract_green.log` (89 passed).
- **`a99eab15` (tests only)** — `eval_FLAC.py` and `depth_canon_cond_probe.py` really start under the launcher
  (`…_13-43-14_production_targets_under_launcher.log`, 2 passed). A characterization test: it passed against the
  existing code, so it has no RED — disclosed.

## 2026-11-05T13:53:09-05:00 — round 5a: the stream registry and the exp35 collect mode, part 1 (`999db221`, `2b129e30`)
- **Registry (`999db221`)** — `exp35_make_stream_registry.py` verifies exp_25's P1-pin / CANON / YAW-pin unseen
  40k T / R / C4 cells offline at `d6ca307a` (84 cells), requires them to agree across arms and freezes input_hash
  per K / seed, assignment_hash per seed (R) and per angle (fixed); 135 / 225 are derived from the split's target
  order + `yaw_column_shift` (192 / 320 columns), and the rule must first reproduce all five observed angles.
  RED `…_13-47-02_r5a_registry_red.log` (collection error) → `…_13-47-37_…_attempt1_test_dirs_fixed_after.log`
  (1 failed: a test reused a tmp dir that held dumps) → GREEN `…_13-48-09_r5a_registry_green.log` (6 passed);
  frozen from the exp-25 worktree: `…_13-48-28_r5a_registry_freeze.log`, sha256 `9ec31f32…`; every prefix equals
  the plan's §5 item 5 table.
- **Part 1 (`2b129e30`)** — the exp_25 results pinned (`ec3fcd08…`) with §5 items 1–2 checked, the registry
  pinned, `registry_reasons`, `--exp35-ctl-check K` (E1; the §3c constants equal the force-added receipts),
  `--exp35-e2` (14 tolerances). RED `…_13-51-09_r5a_collect_part1_red.log` (41 failed) →
  `…_13-52-13_…_attempt1_two_test_bugs_fixed_after.log` (10 failed: a dict built with int keys through `**`, a
  fake E2 result without its `rule`) → GREEN `…_13-52-47_r5a_collect_part1_green.log` (41 passed).

## 2026-11-05T13:57:14-05:00 — round 5b (i): Tier 2-C4 and the Tier 3 B-F row (`0afd2edd`)
- RED `…_13-55-46_r5b_tiers_red.log` (13 failed) → `…_13-56-21_…_attempt1_two_test_bugs_fixed_after.log` (2 failed:
  a 1e-9 perturbation is below float32 resolution, so the "zero control" case saw p99 0 — raised to 1e-6; the
  check list holds 14 metric × K entries, each carrying BOTH rules = the registered 28 checks) → GREEN
  `…_13-56-52_r5b_tiers_green.log` (13 passed).
- **Decisions** — exp_25's registered implementation is kept for B-F: a C4 pair that is not computable counts as
  not passing (`tier2`), Tier 3 is None when an input is missing (`tier3`); B-F's Tier 3 uses an index over the
  BF-pin cells only, so BF-dup never enters; P1-pin's Δ_C4 is read from the pinned JSON at full precision (§11 N7).

## 2026-11-05T14:20:19-05:00 — round 5b (ii): the results pipeline, the page and the CLI (`812d4e38`, `6145f93d`)
- **A first draft was superseded before any commit.** It was written test-first (RED
  `…_13-59-11_r5b_collect_mode_red.log`, 10 failed; then `…_14-00-39`, `…_14-01-23`, `…_14-02-05` attempts — one
  production slip (an unbound `farm`), a fixture "exp_25 pin" that was not d6ca307a, and a missing
  `--no-link-check` in the fixture —
  and GREEN `…_14-03-24_r5b_collect_mode_green.log`, 10 passed; `…_14-04-43_r5b_collector_files_green.log`, 177).
  At commit time it counted +201 code lines (over the rule), and re-reading plan §4 / §5 showed four departures:
  the A6 CSV read outside E2 (§5 item 6), reused exp_25 rows copied into the tables (item 3), a CTL not yet run
  treated as a gap (item 7) and "pending" instead of the s42 one-pin cells on an E2 FAIL (§4). It was set aside
  (scratch `collect_r5b_ii_draft1.py`) and the round redone test-first in two commits.
- **(ii-a) `812d4e38`** — `exp35_results` / `build_exp35_results` / `tables`. RED against the parent collector in a
  scratch clone `…_14-15-01_r5b_ii_a_results_red.log` (19 failed: 18 × no `exp35_results`, 1 × no
  `EXP25_EXTERNAL`) → `…_14-17-29_…_attempt1_test_name_bug.log` (1 failed: the test named the CANON-45 cells
  `…_C4_…`; their registered names are `exp35_CANON_OFF_S40000_K{k}_s{s}_th45`) → GREEN
  `…_14-18-14_r5b_ii_a_results_green.log` (19 passed); the collector test files `…_14-18-42_…` (446 passed).
- **(ii-b) `6145f93d`** — `render_exp35_md` + `--exp35`. RED `…_14-21-46_r5b_ii_b_cli_red.log` (6 failed: no
  `EXP35_RESULTS`, argparse exit 2 on `--exp35`) → GREEN `…_14-22-44_r5b_ii_b_cli_green.log` (6 passed); the
  collector test files `…_14-23-46_…` (452 passed).
- **Real-data smoke (read-only, CPU)** `…_14-25-33_r5b_real_data_smoke.log` (script embedded verbatim): the 24
  external exp_25 cells verify offline at `d6ca307a` against `~/codespace/exp-25-depth-canon`; Δ_R recomputed from
  the external R cells and the JSON's own θ = 0 values equals the JSON's Tier 3 `delta_r`, and the 45-degree probe
  its `probe45`, exactly (max |diff| 0) — the external cells are the ones exp_25 used; `--exp35 --partial` today
  exits 3 with "CTL K1 refused (E1 must be equal)" and writes nothing.
- **Decisions / ambiguities resolved (plan quotes)**
  - A6 CSV — §1 (b) "45°: FLAC and OrbitRIR from the A6 CSV unless E2 fails" vs §5 item 6 "The sha-pinned A6 CSV
    is read only by E2 and the figure." → in the collector E2 is the CSV's only reader; its result carries the
    45-degree rows (`rows45`, filled only on a PASS) and table (b) takes them from there; Tier 2 / Tier 3 never see
    the CSV (§9 "a poisoned CSV leaves Tier 2/3 unchanged" — tested).
  - Reused rows — §5 item 3 "Reused rows stay under `external_reference.exp25` and never pair with H35 cells." →
    table (a) FLAC / CANON and table (c) CANON / P1 / YAW are pointers (`{"ref": "external_reference.exp25…"}`) the
    page resolves; no paired statistic joins an exp_25 and an H35 cell. CANON at 45° pools exp_25 s42 with
    CANON-45 s43–46 in one mean ± sd because §1 (b) prescribes it ("CANON from exp_25 s42 plus CANON-45"); its
    source label says so.
  - E1 — §5 "It refuses unless all of the following hold: … 7. CTL equality holds (§3c)." → "equal" is required
    for any results, partial or not (a CTL not yet run refuses too; the supervisor runs CTL first).
  - E2 FAIL — §4 "FAIL → the +16 phase-(ii) cells run, and 45° comes from one-pin SDPA cells (s42: exp_25 P1-pin
    C4 th45 and exp_35 BF C4 th45)." → table (b) shows those s42 points (n 1), labelled, with a gap; hence the
    external list holds the P1-pin th45 s42 cells too (24 cells). The 16 contingency cells have NO manifest rows
    (open item for the Planner, see the hand-back).
  - Integrity vs gap — an external cell that does not verify, an A6 CSV that is not the pinned one and a B-F probe
    file that is present but not a valid control refuse even under `--partial` (pinned references; §7 lists "an
    invalid probe" as a hard stop); cells not run, a probe not run and an E2 that did not PASS are gaps.
  - Outputs — `--exp35-out PREFIX` (default `<exp_35 folder>/orbitrir_yaw_results`); `--collect`, `--out-json`,
    `--out-md` are refused with `--exp35` instead of being silently ignored; the exp_25 mode is untouched.

## 2026-11-05T14:54:29-05:00 — plan §14 A1: the E2-FAIL block (`4889f43b`, follow-up `753c5285`)
- **Goal** — the Planner's amendment §14 A1 (localization-exp `65651775`), answering my interim report's open item:
  pre-register the 16 E2-FAIL contingency rows ({P1, B-F} × 45° × K1/K8 × s43–46, priority 17, orders 330–345), keep
  them out of every default selection and the supervisor's 33 chunks, and enforce the three E2 cases in `--exp35`.
- **Evidence** — RED `…_14-40-44_a1_e2fail_red.log` (11 failed) → `…_14-42-27_…_attempt1_superseded_reading_test.log`
  (1 failed: the (ii-a) test of my reading (4), replaced by A1.4 — updated) → GREEN `…_14-43-22_a1_e2fail_green.log`;
  the collector / generator files `…_14-44-11_…` (1 failed: the page test's needle for the old label) →
  `…_14-46-08_…_2.log` (468 passed); Codex C3a C7 (any artifact counts as present): RED `…_14-48-29_a1_c7_present_red.log`
  (4 failed) → `…_14-48-56_…_attempt1_test_dump_path_bug.log` (1 failed: the test built the staging path from a fake
  checkpoint name) → GREEN `…_14-50-51_…` (472 passed); the runner by block name `…_14-53-02_a1_runner_manifest_tests.log`.
- **Follow-up `753c5285` (tests only)** — the full runner-file run `…_15-09-53_C3a_H1_runner_files.log` (1 failed, 78 passed)
  showed one exp_25 harness test still expecting the old default selection (every row); my A1 regression run had
  covered only a subset of that file. RED `…_16-04-18_…` → GREEN `…_16-04-24_…`.
- **Decisions** — the E2-FAIL rows take subblock `OFF` (names `exp35_{P1,BF}_OFF_S40000_K{k}_s{s}_th45`, like CANON-45);
  the fallback of §14 A1.4 ("that method's 45° point falls back to the labelled s42-only point") is applied per
  (method, K) — each K is its own 45-degree point; the CANON 45-degree row (always mixed-pin) carries the same per-point
  provenance; an exp_25 point's backend is stated as "sdpa (plan §3b; not in exp_25's schema-1 receipts)".
- **Manifest** — 346 rows; `depth_canon_cells.json` sha256 `440eb1befb34…7037d`; the 116 exp_35 rows: sha256
  `500d6f5ed9c7…252ea` = sha256 of the rows with order >= 230 as compact JSON (`json.dumps(row, separators=(",", ":"))`),
  joined by "\n", plus a trailing "\n".

## 2026-11-05T15:04:05-05:00 — Codex C3a fixes (`a1833f52`, `f4b49808`, `1d8d0ca1`, `a0f6960e`)
- **BLOCKING 1 `a1833f52`** — RED `…_14-55-50_C3a_B1_probe_contract_red.log` (10 failed) →
  `…_14-56-47_…_attempt1_floor_test_made_consistent_after.log` (1 failed: the determinism-floor test edited a raw value
  without rebuilding the summaries, so the probe was refused for the stale summary instead of the floor) → GREEN
  `…_14-57-23_…` (154 passed).
- **HIGH 2 + HIGH 3 `f4b49808`** — RED `…_14-59-23_C3a_H2H3_tier2_red.log` (4 failed) → GREEN `…_14-59-55_…` (51 passed).
  tier2_c4 is tested on synthetic entries (only the dump reader is replaced): the fixture's N = 8 items cannot give a
  rare outlier below 1 %.
- **NIT 1 `1d8d0ca1`** — RED `…_15-01-16_…` (2 failed) → GREEN `…_15-01-40_…` (284 passed with the exp_25 collector tests:
  the exp_25 mode's live bf_historical() is unchanged).
- **HIGH 1 `a0f6960e`** — RED `…_15-03-57_C3a_H1_runner_configs_red.log` (3 failed; the one-digest guard test passed, a
  characterization) → GREEN `…_15-05-24_…` (6 passed) → the runner files `…_15-09-53_…` (78 passed; its one failure was
  the A1 follow-up above). **Ambiguity resolved:** Codex's literal fix ("derive the config paths from the selected rows
  only") would give the P1 and the B-F single-cell chunks different relevant trees, hence different campaign digests,
  and §5 item 4 (one digest) would refuse the campaign; so the pin CHECK follows the selected rows, while the relevant
  tree takes the exp_25 arms' configs for an exp_25-only selection (the d6ca307a set) and every arm's for any exp_35
  selection. A guard test pins the one-digest property.

## 2026-11-05T15:12:43-05:00 — round 6a (i): the training guard (`c58f2808`)
- RED `…_15-10-05_r6a_guard_red.log` (the module absent: the tests had been written first, the guard was moved aside for
  the RED) → `…_15-10-06_…_attempt1_two_test_bugs_one_parse_bug.log` (3 failed: a window count of 23 that is 24; the
  fixture truncated a float sum so 4857 s became 4856 s; a real bug — `parse` created an epoch entry before refusing a
  changed bar total) → GREEN `…_15-10-56_…` (22 passed).
- **Real-log smoke** `…_15-11-18_r6a_guard_real_log_smoke.log` (read-only) found the PL bar reset under the OLD label
  after each validation ("Epoch 0: 0/4550 [00:00<" before "Epoch 1:"), which the fixture lacked: RED
  `…_15-11-49_r6a_guard_reset_red.log` → GREEN `…_15-12-01_…` (23 passed) → smoke 2 `…_15-12-16_…`: epochs 0–5 parse; the
  unshared 100 / 500-step windows are 4.210–4.280 / 4.218–4.244 and epoch 0's worst (his table job) 9.950 / 9.712 —
  the plan's §7 numbers exactly.
- **Decisions** — windows are judged only when they END after the baseline step (our first spawn), so his history is
  never judged against us; the mean is judged from the first step after the baseline, literally (no minimum length);
  a repeat of a step is accepted only for the latest step; the anchor is the 4 KiB before the last line terminator.

## 2026-11-05T16:22:05-05:00 — round 6a (ii)–(iv): the supervisor (`8efdaf5a`, `1307a182`, `fc1647dd`, `0b9e1a36`)
- **(ii) `8efdaf5a`** — RED `…_15-14-54_r6a_supervisor_core_red.log` (the module absent) → GREEN `…_15-15-43_…` (29).
- **(iii)** — the loop tests were written first: RED `…_15-19-06_…_superseded_by_the_corrected_tests.log` (22 failed;
  four test mistakes fixed before any implementation) → RED `…_15-20-25_r6a_supervisor_loop_red.log` (23 failed) →
  `…_15-22-38_…_attempt1_two_test_bugs.log` (2 failed: the probe spawn carries no ALLOW_SHARED_GPU, it does not go through
  the runner; and the table-job test expected a hand-over where §13 B1 registers a latch — split into "during our cell:
  latch" and "while we wait: hand-over") → GREEN `…_16-04-53_r6a_supervisor_loop_green.log` (54 passed). The loop counted
  277 code lines, so it was split: **(iii-a) `1307a182`** lane ownership — its direct tests were written AFTER the
  implementation, for the split (disclosed), RED against the parent `…_16-08-56_r6a_iii_a_lanes_red.log` (10 failed) →
  GREEN `…_16-09-04_…` (39); **(iii-b) `fc1647dd`** the chunk transaction — RED against (iii-a) `…_16-09-46_…` (24 failed)
  → GREEN `…_16-10-03_…` (64).
- **(iv) `0b9e1a36`** — RED `…_16-19-58_r6a_iv_cli_red.log` (14 failed) → `…_16-21-17_…_attempt1_snapshot_followed_symlinks.log`
  (2 failed: the test's file snapshot followed the fake /proc's dangling fd links) → GREEN `…_16-21-48_…` (14 passed).
- **Decisions / ambiguities resolved**
  - The chunk order inside "T s43–46, R, dup, C4+45" (§7): seed-major with K1 before K8; C4+45 angle-major 90, 180,
    270, 45. Planning durations: §12 B1.3's for cells; the probe 10 min from §7's ETA (§12 gives none).
  - "A new process on GPU 1 while our cell runs" (§13 B1) is checked after every poll, including the one at which our
    cell is found finished (a conservative reading: it may have appeared while ours ran); our own processes are the
    runner's /proc subtree; the approved trainer is excluded.
  - A latch never interrupts the running cell: it is watched to its end and verified; its gates are skipped, so it stays
    `runner_verified` for the Planner. A restart refuses while the latch is set; only the Planner clears it.
  - Restart: an in-flight cell is adopted only if its receipt verifies (its rc and end recorded as None, `adopted_at_restart`),
    an in-flight probe always latches, a `runner_verified` chunk re-runs its gates first.
  - The first reservation (GPU 1, epoch 1) is created at the first spawn; a hand-over before any spawn creates GPU 0's
    with epoch 1; at completion lane and reservation are released together.
  - The runner is called with `GPU_MAX_POLLS=1`, so a lane refused at its own final check is an rc 2 (the latch), never a
    silent wait; the supervisor waits instead, polling every 60 s (5 min while a cell runs).

## 2026-11-05T16:18:48-05:00 — Codex C3b fixes (`f5a11e2a`, `526308c8`)
- **HIGH 1 + NIT 1 `f5a11e2a`** — RED `…_16-10-37_C3b_H1N1_red.log` (10 failed) → GREEN `…_16-11-05_…` (63 passed).
- **BLOCKING 1 + HIGH 2 `526308c8`** — RED `…_16-13-58_C3b_B1H2_state_reservation_red.log` (58 failed: the new pins in the
  signatures plus the missing checks) → GREEN core + lanes `…_16-15-48_…` (79) → all supervisor tests `…_16-16-09_…` (104).
  Loop tests updated to the exact schema (a complete runner_verified record; the hand-over test reads the last chunk's lane
  because lane and reservation are released together at the end).

## 2026-11-05T17:20:00-05:00 — the verification baseline at `b15d8a82`, and the Codex C3c fixes (`dce4f27a`)
- **Baseline at `b15d8a82`** (C3c judged it not H35; kept as a baseline):
  - all kit suites IN the worktree `…_16-26-08_final_all_suites_in_worktree_at_b15d8a82.log`: **1420 passed, 1 skipped, 0
    failed** (the autolaunch test that fails only in a /tmp clone passes here);
  - static checks `…_16-26-50_final_static_checks_at_b15d8a82.log`: whole-repo `--collect-only` 6421 tests, 0 errors;
    `git diff --check` clean; `eval_FLAC.py` / `src` (minus tests) untouched since localization-exp; since d6ca307a only
    `src/localization/*`; the exp_25-only render `1073ef64…` byte-identical to d6ca307a's file;
  - the exp_25 byte guard, plan §5 command verbatim from the worktree `…_16-26-50_final_exp25_recollect_byte_guard.log`:
    `acb15013…`, NOT `ec3fcd08…` — two REPO-relative inputs fail in any checkout but the exp-25 worktree: the CANON probe's
    provenance names its model config by ABSOLUTE path into the exp-25 worktree (the check compares realpaths with THIS
    checkout's config), and the tie audit re-reads 9,650 depth maps under `REPO/AcousticRooms` (this worktree has no asset
    link, per the brief). In a scratch clone of `b15d8a82` with those two inputs made equivalent (`AcousticRooms` -> the
    same NAS path as the exp-25 worktree's link; `FLAC_AR_CANON.json` -> the exp-25 worktree's byte-identical file)
    `…_16-37-00_final_exp25_recollect_byte_guard_equivalent_env.log`: **`ec3fcd08…` exactly, md identical** — the exp_25
    mode is unchanged byte for byte; the difference was the environment only (an ambiguity of the plan's command, which
    assumed those inputs resolve in the run worktree);
  - the runner DRY_RUN over the 32 phase-(i) cells `…_16-28-25_…` (32 would run, the pre-flight OK, rc 0, GPU processes
    unchanged) and the supervisor's 32 per-cell commands as dry runs in the §7 order `…_16-30-02_…` (32 rc 0, ONE campaign
    digest `9cd502af…` — the one-digest property on the real tree); their artifacts moved to `dry_run_artifacts_b15d8a82/`;
  - the supervisor's live read-only dry run `…_16-27-54_…` (run with `-B`): DRY RUN OK, identity OK ([3028196]), guard
    OK, the reservation directory untouched;
  - bytecode: my first runner DRY_RUN wrote 13 `.pyc` under `__pycache__/` and `src/**/__pycache__/` (the runner calls
    the collector without PYTHONDONTWRITEBYTECODE; none under the overlay) — removed, `…_17-07-22_bytecode_cleanup.log`;
  - his 100-step windows over epoch 5 stayed 4.22–4.34 s/step during these runs (one window at 4.34 around 16:45, while
    two NAS-heavy runs overlapped — disclosed; the authoritative ladder runs the NAS-heavy byte guard alone).
- **C3c fixes `dce4f27a`** — BLOCKING 1 + HIGH 1: RED `…_16-39-00_C3c_B1H1_red.log` (60 failed) → GREEN
  `…_16-42-05_C3c_B1H1_green.log` (133); HIGH 3 + HIGH 4: RED `…_16-45-16_C3c_H3H4_red_hung_stopped_no_horizon.log` (the
  PID-reuse test made the unfixed supervisor wait for ever for a busy GPU 0; I stopped my own run and gave the fake world a
  sleep horizon) → RED `…_16-56-16_C3c_H3H4_red.log` (6 failed) → `…_17-06-56_…_attempt1_restart_test_fake_gates_without_probe_json.log`
  (1 failed: a test fixture) → GREEN `…_17-10-28_C3c_H3H4_green.log` (140); HIGH 2: RED `…_17-10-39_C3c_H2_dry_run_red.log` (5 failed) + the defect demonstrated
  `…_17-11-14_C3c_H2_bytecode_demo_red.log` (the committed dry run wrote two `.pyc` in a fresh snapshot while printing "nothing was
  written") → `…_17-13-49_…_attempt1_two_test_bugs.log` (214 passed, 2 test bugs) → GREEN `…_17-17-29_C3c_H2_cli_green.log` (20) and every supervisor / guard /
  identity file (214 + the two fixed tests).
- **Decisions** — a vanished approved trainer is not a replacement (an unreadable /proc entry = gone; a reuse is caught
  at the next poll or the pre-spawn check); the probe ledger is checked by the supervisor itself (the sidecar seals the
  JSON; the launch record names this log, its sha256 and the campaign digest).

## 2026-11-05T19:06:20-05:00 — the Codex C3d fixes (`955ac38c`, `d3381fed`) and Planner P1
Codex C3d (`orbitrir_yaw_evidence_codex_code_C3d_review.md`, main checkout) = REQUEST-CHANGES, 3 HIGH, all in
`exp35_supervisor.py`; the Planner accepted Codex's fixes and added P1 (a single-instance lock).
- **HIGH 1 + HIGH 2 `955ac38c`** — RED `…_18-32-50_C3d_H1H2_red.log` (49 failed, 148 passed; the unreadable-PID tests
  ran the unfixed supervisor into the fake world's 600-sleep horizon) → `…_18-44-02_C3d_H1H2_green_attempt1_test_bug_truncate_before_read.log`
  (5 failed: my test helper truncated a reservation file before reading it, `open(p, "w").write(json.load(open(p)))`) →
  GREEN `…_18-48-51_C3d_H1H2_green.log` (197 passed: the four supervisor files).
- **HIGH 3 + P1 `d3381fed`** — RED `…_18-53-58_C3d_H3P1_red.log` (7 failed, 159 passed) → GREEN `…_18-59-29_C3d_H3P1_green.log`
  (264 passed: the probe-contract file + the four supervisor files) and `…_19-05-12_C3d_H3_collector_files_green.log`
  (183 passed: every other `test_exp35_collect*` file, the guard and the identity helper).
- Regression guards that pass before AND after (disclosed, not RED): a trainer that exits between two observations
  (listed, unreadable, absent from the re-queried table) is gone -- no latch, the hand-over follows (lanes + loop); a
  second finalization of any OTHER record is refused and writes nothing (6 cases); a restart with a changed ledger
  latches (the log appended; the JSON re-sealed in the finalizer's own format -- this one needs the supervisor's ledger
  check once the finalizer is idempotent).
- **The four rules (for the launch entry):**
  1. *Unreadable PID:* a PID nvidia-smi lists on our lane whose complete `/proc` identity cannot be read is re-checked
     against a freshly queried compute table -- gone only if it is no longer listed; still listed (or the table cannot
     be re-read) = the stop latch, never "approved" (the running-cell watch: the cell finishes, never signalled;
     `ready()`: no spawn, no hand-over).
  2. *Authenticated transfer / release:* the state holds the reservation as `{uuid, epoch, path, sha256}` (the sha256
     of the exact bytes written); it is authenticated (sha256, then field by field, the epoch) before every lane
     decision, before and after the 60-s hand-over gap, right before the hand-over commit (the old and the new
     record), right before every removal, and after the in-flight save as the LAST step before a spawn (a spoiled
     record there spawns nothing and returns the chunk to planned); a removal unlinks only the authenticated file,
     through the inode just read; any mismatch latches and unlinks nothing.
  3. *Probe restart:* from chunk 5's persisted post-finalizer / pre-final-guard state a restart re-runs the gate; the
     collector's finalization is idempotent only when the JSON is its own bytes, sealed by its sidecar, with a launch
     record exactly equal to the one it would write (backend, overlay + launcher sha256, campaign digest, log path,
     the CURRENT log sha256) -- then nothing is written and the contract re-runs; the supervisor also requires the
     re-computed JSON / sidecar / log sha256 to equal the ledger it persisted; then the final guard; anything else latches.
  4. *Single instance:* `Supervisor.run()` takes `flock(LOCK_EX | LOCK_NB)` on `<state>.lock` before reading the state
     and holds it until the run ends; held -> `SUPERVISOR REFUSED: another supervisor holds <lock>`, rc 3, the state
     untouched; `--dry-run` never creates the lock (an existing one is probed with `LOCK_SH | LOCK_NB`, informational).
- **Decisions / readings** -- HIGH 3 by Codex's first option (an idempotent finalizer; the exp_25 mode never calls it)
  plus the persisted-ledger equality; HIGH 1's "complete fields" as the sha256 of the exact bytes (STATE_SCHEMA stays
  1: the `supervisor_sha256` pin already binds any state to this exact code); the first-spawn baseline now set after
  the reservation step (a refused first reservation used to leave a baseline without a started chunk -- an unloadable
  latched state); the re-query happens at once, no delay.
- **Residuals (disclosed)** -- POSIX has no compare-and-unlink: between the inode check and the unlink there remain
  microseconds; a trainer caught mid-exit (its fds closed, its CUDA context not yet released) and still listed at the
  re-query latches (fail-closed; a short re-query delay would reduce that, not added); the dry run's `LOCK_SH` probe
  could make a supervisor starting in that same microsecond refuse (rc 3, nothing touched); a crash between the
  hand-over commit and the GPU 1 removal still latches at the restart (both files on disk; unchanged).
- The authoritative ladder runs at this records commit; its transcripts stay untracked (H35 = this HEAD).
