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
