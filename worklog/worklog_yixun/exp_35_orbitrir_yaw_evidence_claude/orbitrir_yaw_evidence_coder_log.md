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
