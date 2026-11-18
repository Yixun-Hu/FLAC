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
| (this commit) | 2026-11-18 03:08 | records: Q1, the first (ABORTED) and the final L0 run, the U logs; worklog entries; open item 13 | records | Q1 (727 passed, 5 skipped); **final L0: 2,522 passed, 15 skipped, 41 deselected, 0 failed at `e6920ace`, tracked-clean** |
