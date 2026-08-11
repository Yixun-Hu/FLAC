# Unbundling ruling — q9 before restart-leg records (pin 89f24cd)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-10

# RULING: ACCEPTABLE — Q9 RESULTS NEED NOT BE QUARANTINED

The missing restart-leg records are irrelevant to q9’s INITIAL-only 40k checkpoints. All 20 q9 cells are transaction-bound to `source_sha=89f24cd`, which contains every review fix and the audited C32 anchor.

Conditions:

- Publish q9 only after all 20 cells validate and `check_q9_round` passes at `89f24cd`.
- Submit no `traj` cells until validated restart-leg records are committed and the campaign receives its one further pin bump.
- Record this two-pin unbundling and the extra pin move in the worklog.
