# commits_yaw_gen — exp_14 commit ledger

Base commit: `89f24cd` (branch `check-equivariance-necessity`; other-agent exp_11 anchor commit rebased onto `f579777`).

| SHA | Round | Description |
|---|---|---|
| `66e6ca5` | r1 | STEP-0 fixed-mode byte-compat contract (golden fixtures captured at `44788e6` + snapshot pins) |
| `9e737a1` | r1 | `draw_yaw_offsets` / `offsets_to_radians` (TDD cycle 1) |
| `ebd7983` | r1 | rotation-plan resolution + injective `_rotrand<seed>` naming (TDD cycle 2) |
| `dbf0fae` | r1 | §3.3 assignment-integrity stream + canonical hashes (TDD cycle 3) |
| `16d7d13` | r1 | random-yaw eval path wired end to end (TDD cycle 4) |

| `a6d2e26` | r1-fix | F4 — context-fingerprint dtype/shape/finiteness pins (review N4) |
| `769710e` | r1-fix | F3 — per-position `idx == i` substitution assertion (ruling 3) |
| `59be9ff` | r1-fix | F1 — `--expected-stream-count` ends the tautological count check (review B1) |
| `6efbcb7` | r1-fix | F2 — opt-in `.stream.json` full-payload sidecar (review B2, ruling 1) |
| `6e7616c` | r1-fix | F5 — multi-worker loader ordering test (review N5) |

**Round 1 CLOSED 2026-08-11T01:02 EDT** — Codex re-verify `yaw_gen_codex_code_r1_reverify.md`: all findings confirmed closed; suites independently rerun (134 passed).

| `b45fe21` | r2 | COMMIT A — verbatim copy of the exp_11 screen kit (blob SHAs in message, `cmp`-verified) |
| `bcd5027` | r2 | sbatch delta 1/5 — namespace, paths, campaign constants |
| `037de5d` | r2 | sbatch delta 2/5 — cell contract: {rgen,zref,vctl} × 5 arms, STEP=40000 only |
| `74ebf16` | r2 | sbatch delta 3/5 — eval names, log names, protocol banner |
| `f15bd20` | r2 | sbatch delta 4/5 — lineage gates: exp_11 registry read in place |
| `139a36c` | r2 | sbatch delta 5/5 — eval argv, protocol manifest, per-cell validation |
| `d4736c6` | r2 | `exp14_validate_cell.py` + TDD suite (74 tests) |
| `a00f1ba` | r2 | single-cell submitter — exp_14 contract, own campaign pin |
| `681ddf8` | r2 | `yaw_gen_submit_grid.sh` — wave submitter, validate-before-skip dedup |
| `05e6c6d` | r2 | guard suite — exp_14 contract end to end (162 cases) + run logs |

Round 2 interrupted once by the account session limit (resumed post-4am reset, completed 12:39 EDT). Interleaved non-exp_14 commits in the range belong to the concurrent session (exp_15 r1, exp_11 q9/restarts, exp_10 A4).

| `5d6e349` | r2-fix | B1 — rgen/zref pass no `--rotate-deg`; tolerant `parse_deg`; single `check_argv` |
| `98e276c` | r2-fix | B6 — six fail-closed validator gaps closed |
| `06b66b0` | r2-fix | B4 — dedup rests on checkpoint identity; `exp14_ckpt_expect.json` (Cn digests == exp_11 registry; VANL established) |
| `1ab22a0` | r2-fix | B2 — Slurm job names carry the rotation token (cell-injective over all 106) |
| `a201e3c` | r2-fix | B3+B5+B7+NIT8a/b — pin-file rail, lease-verified in-flight, atomic manifests, full-grid path pins, single-cell DRYRUN |
| `0f056b4` | r2-fix | NIT8c — live-wave guard cases (guard suite 162→180) |
| `6e6f437` | r2-fix2 | X1+X2+X3 — seams honored only on provably-non-submitting paths; durability traps + sentinel manifests; single-source check-argv. Includes the ⚠️ incident response (4 unplanned PENDING jobs cancelled at 00:00:00 elapsed; command log scrubbed of mock lines + annotated). *(SHA was `135cb4b` pre-rebase)* |
| `429b871` | r2-fix3 | Y1+Y2 — test mode simulates Slurm internally; hold guarded from the first instruction (`scancel --name` fallback, nonzero re-raise). *(was `2131cfb` pre-rebase)* |
| `874d110` | r2-fix4 | Z1–Z4 — no env var names a command in any mode (per-mode allowlists, fixture-file data only); canonical absolute-path Slurm resolution with `unset -f`; guard live-case isolated in temp MAIN_REPO; suite-end byte-identity EXIT trap; committed pre-fix RED proof (8/8, recording stand-ins) |
| `d256014` | r2-fix5 | W1–W4 — native env enumeration + gate-first ordering; probe isolation; sha-or-ABSENT EXIT assertions + suite_rc; stale lease `7654321` removed via exp_11 helper (evidence); redproof contained to temp store |
| `a23b551` | r2-fix6 | V1–V3 — unshadowable `POSIXLY_CORRECT` preamble; redproof post-fix probes retargeted (whole-store 29-lease containment check); isolation-by-default via suite-global `YAW_GEN_MAIN_REPO`; threat-model boundary written into scripts/harnesses |
| `d200e1b` | r2-fix7 | U1–U3 — sweep above `set -euo pipefail` (shadowed-set() probe); `iso_off`/`iso_on` push-pop opt-outs + `assert_isolated`; command-position eval/sbatch lint (+ twin sbatch-lint fix, disclosed) |

**Round 2 CLOSED 2026-08-11T17:40 EDT** — closure basis: Codex checks #1–#6 (all findings either confirmed-closed or resolved by the final three mechanical diffs) + Planner verification of the U-batch (preamble order read directly; bash -n ×5; DRYRUN=106/nothing-submitted; pytest 254; queue 0). Guard suite 218/0 `suite_rc=0`; redproof 10/0 contained. Threat-model boundary: accidents/stray state (defended); adversarial shell environments (recorded, out of scope) — worklog 16:55 entry.

Note: effective base drifted `89f24cd` → `44788e6` (concurrent session's worklog-only commit landed first; `eval_FLAC.py`/`yaw_rotation.py` byte-identical at both). `809ece5` interleaved in the range is the exp_15 scaffold (other session, worklog-only).

## Round 3 — collector (plan §5.6) + model_comparison contract (plan §5.7, amended)

| SHA | round | subject |
|---|---|---|
| `24fea16` | r3 | collector core — artifact parsing, provenance (validator IMPORTED), §3.3 equalities, 5/5 blocks |
| `170aeee` | r3 | the §4 estimation conventions — paired-t (t₀.₉₇₅,₄ as a constant), Holm, metric directions |
| `10aa981` | r3 | gates G1–G4 (blocking) + the G5 check (never blocking); per-angle assignment grouping fix |
| `f874758` | r3 | readouts, rendering, CLI, JSON bundle + four self-test transcripts |
| `825b7fc` | r3 | `gen_model_comparison.py`: ten exp_14 Z rows behind their own `exp14z` contract |

Round-3 notes:

- **exp_11 specs verified byte-untouched**: the exp_11 row-spec fingerprint is
  identical loading `HEAD`'s `gen_model_comparison.py` and the new one —
  `(12 rows, 57820d20d7a4…)` on both sides — and `test_exp11_row_specs_are_byte_untouched`
  pins that digest going forward. Total ROWS 64 → 74 (+10 exp_14 Z rows).
- **The published `model_comparison.md` was deliberately NOT regenerated.** Verified
  instead against the real evidence tree through a symlinked sandbox root: the ten new
  rows render `*pending (0/5 seeds on disk)*`, every existing row is byte-identical, and
  the generator exits 0. Regeneration + push is the Planner's §6 sequencing step when the
  VANL Z two-K × five-seed transaction completes.
- **Rebase refused** (as the round-3 brief anticipated): the working tree carried the
  OTHER session's uncommitted `exp_15` worklog edits, so `git pull --rebase` aborted with
  "cannot pull with rebase: You have unstaged changes". Their files were never stashed,
  cleaned or checked out; all five round-3 commits are local-only and unpushed.
