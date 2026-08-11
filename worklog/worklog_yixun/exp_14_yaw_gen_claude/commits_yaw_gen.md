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

Note: effective base drifted `89f24cd` → `44788e6` (concurrent session's worklog-only commit landed first; `eval_FLAC.py`/`yaw_rotation.py` byte-identical at both). `809ece5` interleaved in the range is the exp_15 scaffold (other session, worklog-only).
