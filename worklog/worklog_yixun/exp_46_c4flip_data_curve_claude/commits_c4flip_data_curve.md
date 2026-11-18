# exp_46 c4flip_data_curve — commit ledger

Branch `exp-46-c4flip-data-curve`, base `X45f = 1e37ead2a352bd78d50cdda9ed02e95738a4ad04` (exp_45's frozen tip).
Test logs live in this folder as `c4flip_data_curve_<YYYY-MM-DD_HH-MM-SS>_pytest_<label>.log` (every pytest run in a
CPU Slurm job; a failed attempt keeps its log, renamed `..._ABORTED_<reason>.log`). "Changed lines" = added + deleted
(`git show --numstat`); commits above the SOP's 200 lines are marked **>200** with the reason.

| SHA | Time (EST) | Content | Changed lines | Test logs |
|---|---|---|---|---|
| `1a867274` | 2026-11-17 13:45 | records (coordinating session): open exp_46; query log, plan r0, worklog, ledger (no code) | 261 (records) | — |
| `505f75b9` | 2026-11-17 14:10 | records: Codex plan r1 (REVISE, 11 findings) and plan r1 | 403 (records) | — |
| `d77ba58b` | 2026-11-17 14:28 | records: Codex plan r2 (REVISE, 6 findings) and plan r2 | 188 (records) | — |
| `d379f9b2` | 2026-11-17 14:29 | records: corrected times; Codex plan r3 prompt | 72 (records) | — |
| `a3511f5f` | 2026-11-17 14:50 | records: Codex plan r3 (APPROVE-WITH-CHANGES) and plan r3 (final) | 55 (records) | — |
| `99d5390f` | 2026-11-17 15:20 | records: rebased onto X45f; SHA mapping | 13 (records) | — |
| `73cbe268` | 2026-11-17 22:21 | records: plan r3 approved by Yixun (M1–M4 as recommended); L0 starts | 28 (records) | — |
| `01b10973` | 2026-11-17 22:35 | L0 §3.1 T1: the six C4F training-path tests (`test_exp46_c4flip_factory.py`, `test_exp46_c4flip_wrapper.py`), red before the source change | 710 **>200** (two new test files, not flagged in the message) | D0 / D0b (exp_45 node inventory), T1 red |
| `f796fe81` | 2026-11-17 22:53 | L0 §3.1 T2: remove exp_43's flip + offset_step scope refusals (`factory.py`, `diffusion.py`); declared edits E1–E3 | 40 | T2a red, T2b green (C4F + E1–E3 + 28 training-path files; 11 failed + 40 errors all `wandb 0.26.1 != the pinned 0.15.4`, environmental, see worklog) |
| `407d11d1` | 2026-11-17 23:09 | L0 commit A: verbatim rename-copy of the nine exp45_* kit files and the 11 kit-behaviour test files + fixtures at X45f; `test_exp46_provenance.py`; node-level disposition in the message | 12,945 **>200** (a verbatim copy, not flagged in the message) | A1 red, A2 green |
| `942db750` | 2026-11-17 23:27 | L0 B1: rename to exp_46's names; split the mixed test files (H / B nodes out) | 1,356 **>200** (mechanical rename + node split; flagged) | B1a red, B1b ABORTED (compare skip id renamed), B1c green |
| `10c4200f` | 2026-11-17 23:37 | L0 B2: names / paths — exp_46 snapshot roots, A6000 SMOKE_DDP refused, run-dir internals; the path-literal sweep | 331 **>200** (the sweep's registrations; not flagged in the message) | B2a red, B2b ABORTED (sweep unregistered), B2c listing, B2d green |
| `dfbbda54` | 2026-11-17 23:46 | L0 H: the RF-100 bridge removed (eval, collector, table, closure, fixtures) | 860 **>200** (a removal; flagged) | H1 red, H2 ABORTED (row count, needle), H3 green |
| `2ca672ae` | 2026-11-18 00:09 | L0 C: the C4F arm — config (sha pinned), ARMS / check_config / banner gate, run ids, C4F disclosures | 855 **>200** (RF→C4F test renames + the 202-line config; flagged) | C1 red, C2 / C3 / C4 ABORTED, C5 green |
| `1771258b` | 2026-11-18 00:15 | L0 D: the closure admits the C4F config and the f100 delta manifest by exact path | 106 | D1 red, D2 green |
| `ce39a91d` | 2026-11-18 00:29 | L0 E: FRACS "100" and the four runs; f100's audit / context-policy binding; G7 loader-length test | 267 **>200** (not flagged in the message: kit 9 functions + tests) | E1 red, E2 ABORTED (three-run test assumptions), E3 green; G7 confirmation job |
| `90292adf` | 2026-11-18 00:49 | L0 F: f100 delta content manifest (new `exp46_content_manifest.py`) and its in-job re-verification before sealing | 645 **>200** (new tool 177 + new test 302; flagged) | F1 red, F2 ABORTED (eval world f100 pin), F3 green |
| `1775f0f0` | 2026-11-18 00:49 | L0 E4: exp_45's merge-provenance audit at fixed X45f endpoints (declared edit E4) | 20 | E4a red, E4b green |
| `c9e82921` | 2026-11-18 01:03 | L0 G: isolation from exp_38's AND exp_45's roots before every write at every mutating entry point | 257 **>200** (the 42-node matrix; flagged) | G1 red, G1b red (continue), G2 green (2 pre-existing storage failures from E, fixed in P) |
| `ced65850` | 2026-11-18 01:07 | L0 P: the storage tests' reservation arithmetic for four runs (follow-up to E) | 20 | P1 red, P2 green |
| `d858b53b` | 2026-11-18 01:16 | L0 S: C3 — exp_45's reservation ledger read read-only by the submitter and the FULL start | 228 **>200** (C3 tests 182; flagged) | S1 red, S2 green |
| `26d644d0` | 2026-11-18 01:25 | L0 J: the exp_46 table's 20 rows | 54 | J1 red, J2 green |
| `c6c1c197` | 2026-11-18 01:34 | L0 K: the generator's exp_46 contract (8 rows; exp_45 dormant); declared edit E5 | 432 **>200** (contract 148 + new test 251; flagged) | K1 red, K2 green (every generator test file; no E6) |
| `cb8a7850` | 2026-11-18 01:35 | L0 M: the L1 regression policy extended (E1–E5, new exp_46 nodes, exp_46 env skips, REPLACED) | 156 | M1 red, M1b red (continue), M2 ABORTED (base count), M3 green |
| `1036bb0b` | 2026-11-18 01:37 | L0 N: the exp_46 provenance audit (allowed paths X45f..HEAD, content checks, the Y46 E0 rule) | 259 **>200** (one audit test file; not flagged in the message) | N1 green (audit of existing history: no red) |
| `c50855b6` | 2026-11-18 01:37 | L0 O: the registered init-digest prediction (exp_38's values, read never retyped; fail-closed evidence) | 80 | O1 green (with N1; no red) |
| `76a2a81c` | 2026-11-18 01:45 | records: this ledger, the L0 worklog entries (disposition, G7, open items), every finished L0 log (D0 … O1) | records | — |
| `e6920ace` | 2026-11-18 02:23 | L0 U: test isolation from lightning's global rank — the C4F banner test pins rank 0; exp_46's sampler copy restores the rank (exp_45's original leaks it) | 9 | first final L0 run ABORTED (16 failed: the rank leak), collect-only node map, U1 ABORTED (not the polluter), U1b red, U1c red, U2 green |
| `e0cdea5a` | 2026-11-18 03:08 | records: Q1, the first (ABORTED) and the final L0 run, the U logs; worklog entries; open item 13 | records | Q1 (727 passed, 5 skipped); **final L0: 2,522 passed, 15 skipped, 41 deselected, 0 failed at `e6920ace`, tracked-clean** |
| `671ae6c0` | 2026-11-18 03:31 | records (coordinating session): Codex code round 1 (REVISE: 1 HIGH, 9 MEDIUM, 2 NIT) -- prompt, review, log; the coordinator's L0 verification and dispositions | 174 (records) | — |
| `b3932b53` | 2026-11-18 03:47 | fix r1 F1 (HIGH): the manifest tool and the supervisor disable bytecode writes before any import (no `__pycache__` in exp_38's folder) | 58 | V1 red, V2 green |
| `af287964` | 2026-11-18 03:58 | fix r1 F2: the isolation guard at the auxiliary writers (manifest `--build`, init digest `--json`, L1 `--out`, `snapshot_tools()`), in the 42-node matrix | 56 | V3 red, V4 green |
| `e4027a76` | 2026-11-18 04:03 | fix r1 F4: the two f100-pin tests inject the unpinned None (green before and after the L2 pin) | 7 | V5 red (stand-in pin), V6 green (stand-in pin), V7 green (unpinned) |
| `ce849135` | 2026-11-18 04:08 | fix r1 F3: a pending TERM after the last verified read leaves the f100 FULL unsealed; the commit boundary | 45 | W1 red, W2 green |
| `b23cfae6` | 2026-11-18 04:13 | fix r1 F5: L3 compares the measurement with the registered prediction; exit 1 on any disagreement | 94 | X1 ABORTED, X2 ABORTED (test bugs), X1b red, X3 green |
| `8b3e60a2` | 2026-11-18 04:19 | fix r1 F7: C3 tells genuine absence from stat errors; exp_45's whole ledger schema validated | 82 | Y1 red, Y2 green |
| `f22ae572` | 2026-11-18 04:20 | fix r1 F8: the 20-row contract in `build()` and at the renderer boundary, tested at the entry points | 86 | Z1 ABORTED (test bug), Z1b red, Z2 ABORTED (message), Z3 green |
| `7d5bd63c` | 2026-11-18 04:23 | fix r1 F9: the Y46 audit checks each E0 commit whole (exactly the 24 added files) and closure(Y46) == closure(X46) (`closure_digest_at`) | 130 | AA1 red, AA2 green |
| `0c107125` | 2026-11-18 04:30 | fix r1 F11 (NIT): the stream oracle's step seed is X45f's own counter | 47 | AB2 green (test-only; no red) |
| `e2003dc2` | 2026-11-18 04:30 | fix r1 F12 (NIT): the supervisor documents no A6000 training mode; an A6000 SMOKE is refused first | 31 | AB1 red, AB2 green |
| `86425120` | 2026-11-18 04:35 | fix r1 F10: both command sheets (neuronic; A6000 evaluation and publication only) + `test_exp46_command_sheets.py` | 492 **>200** (230 lines of sheets + the 262-line test; not flagged in the message) | AC1 red (sheets held aside), AC2 green |
| `5286f6a6` | 2026-11-18 04:35 | fix r1 open item 1: the wandb-environment failures are inherited only as the parent's exact node / outcome / signature; no deselection carried forward (test; the policy already did it) | 22 | AC2 green (audit: no red) |
| `c8a04475` | 2026-11-18 04:36 | records: fix round 1 -- ledger, worklog entry, every fix-round log | records | — |
| `9cda0203` | 2026-11-18 05:21 | records: the fix round's final targeted run | records | **FR1: 2,579 passed, 15 skipped, 41 deselected, 0 failed at `c8a04475`, tracked-clean** |
| `4b5efd90` | 2026-11-18 05:37 | records (coordinating session): Codex code round 2 (REVISE: 2 HIGH, 7 MEDIUM) -- prompt, review, log; the coordinator's verification of fix round 1 | 123 (records) | — |
| `688daea8` | 2026-11-18 05:55 | fix r2 R2-3 (F2): the init digest's `--json` is made absolute before the cwd change; one path checked, re-checked and written (O_EXCL \| O_NOFOLLOW) | 55 | AD1 red, AD2 green |
| `bf11bc65` | 2026-11-18 06:01 | fix r2 R2-4 (F8): `render()` re-derives every row from its admitted source (the sealed import record; the offline-admitted C4F cells of the E0 receipt's campaign) | 79 | AE1 red, AE2 green |
| `78bd15cf` | 2026-11-18 06:25 | fix r2 R2-1 part 1 (HIGH): `check_destination` and the guarded primitives; the atomic writers validate destination and temp name (O_EXCL \| O_NOFOLLOW); `allow=` for the registered exp_38 import exception | 308 **>200** (code 135 + test 173; flagged) | AF1 red, AF2 ABORTED (write_sealed's exists message pre-empted the guard), AF3 green |
| `26908fee` | 2026-11-18 06:39 | fix r2 R2-1 part 2: the registry's mutations (ledger lock / ledger / probe, environment receipt + lock, SMOKE receipt, `snapshot_tools`, every run-dir writer) | 146 | AG1 red, AG2 ABORTED (a test wrapper's signature), AG3 green |
| `29057aba` | 2026-11-18 06:49 | fix r2 R2-1 part 3: the submitter's (every destination before the first write) and the supervisor's mutations | 191 | AH1 red, AH2 ABORTED (a test's submission log outside the exp_46 folder), AH3 green |
| `b7c8a0e9` | 2026-11-18 07:01 | fix r2 R2-1 part 4: the evaluator's, the collector's, the table's and the auxiliary writers' mutations | 127 | AH1 red, AI1 ABORTED (a test spy's signature), AI2 green |
| `0f1b9933` | 2026-11-18 07:04 | fix r2 R2-9: abandon refuses a sealed run and names `release --run` | 40 | AJ1 red, AJ2 green |
| `d909632b` | 2026-11-18 07:07 | fix r2 R2-8: `exp46_nas_publish.py` -- exclusive staging before rsync, no-replace publication (renameat2 RENAME_NOREPLACE; checked fallback) | 250 **>200** (helper 108 + test 142; not flagged in the message) | AK1 red, AK2 ABORTED (a final link refused by the alias check first), AK3 green |
| `e80a88f8` | 2026-11-18 07:09 | fix r2 R2-2 (HIGH), R2-5, R2-6, R2-7 and the sheet parts of R2-8 / R2-9: both command sheets | 434 **>200** (sheets 173 + test 261; flagged) | AL1 red, AL2 green |
| `b3027add` | 2026-11-18 07:11 | fix r2 R2-1 part 5: the AST audit of every mutating call of the kit (21 audited raw calls) | 98 | AM1 red (the kit before part 1), AM2 green |
| `5827fe08` | 2026-11-18 07:13 | records: fix round 2 -- ledger rows, the worklog entry with the mutation-site audit list, every fix-round-2 log | records | — |
| `fa6b8eee` | 2026-11-18 07:57 | records: fix round 2's final targeted run | records | **FR2: 2,928 passed, 15 skipped, 41 deselected, 0 failed at `5827fe08`, tracked-clean** (neu308, 40 min 51 s) |
| `c8971847` | 2026-11-18 08:20 | records (coordinating session): Codex code round 3, FINAL (REVISE: 1 HIGH, 6 MEDIUM; E7 pre-review); review cap reached | 154 (records) | — |
| `9d304451` | 2026-11-18 08:46 | fix r3 R3-1 (HIGH; closes R2-1 / F2): `resolve_dotdot` -- a '..' after a symlink refuses; every primitive passes the VALIDATED path to the syscall | 187 | AN1 red, AN2 ABORTED (refusing every '..' broke R2-3's relative path), AN3 green |
| `8078440b` | 2026-11-18 08:55 | fix r3 R3-2: outputs_FLAC made / reused as a CONTAINER; the first eval snapshot through the real guards | 48 | AO1 red, AO2 green |
| `72a2b98a` | 2026-11-18 09:00 | fix r3 R3-4 (closes R2-8): no fallback rename -- renameat2 RENAME_NOREPLACE (wrapper or raw syscall) or refuse; the audit entry removed | 126 | AP1 red, AP2 green |
| `f71241e2` | 2026-11-18 09:02 | fix r3 R3-7 (kit): `exp46_nas_publish.py provision` -- the exp_46 checkpoint and dump roots, mount-checked, created through the guard | 118 | AQ1 red, AQ2 green |
| `6180d8ef` | 2026-11-18 09:06 | fix r3 R3-5, R3-6 and the sheet part of R3-7 (closes F10): completion barriers, set -eo pipefail, explicit lane waits, provisioning lines; gate tests that execute the sheets | 271 **>200** (flagged; the message says 238 -- the sheets are 98 and the test 173 lines) | AR1 red, AR2 ABORTED (two tests named old command forms), AR3 green |
| (prepared, NOT committed on this branch) | 2026-11-18 09:08 | D1 / R3-3: the E7 + E8 provenance rule, their L1 registration, the D1 rank-isolation regression and probe -- kept on the local branch `wip-exp46-d1-e7-e8` (see the worklog) | — | AS1 (expected red: the 4 checks that need E7 + E8 themselves; 42 passed) |
| (this commit) | 2026-11-18 09:15 | records: fix round 3 -- ledger rows, the worklog entry (each fix "fixed without re-review (review cap)"), every fix-round-3 log | records | — |
