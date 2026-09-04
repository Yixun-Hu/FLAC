# exp_22 — ORBITRIR frame-averaging port — analysis

**By-line:** Claude Fable 5 (Planner/analyst); Opus 5 max-effort subagents coded all rounds; OpenAI Codex `gpt-5.6-sol` xhigh reviewed (4 rounds + plan review + closure re-verify).

## Reliability judgment: the port is CORRECT, with unusually strong evidence
1. **Bit-level behavioural identity.** Both acceptance cells reproduce the registered B-F 40k rows to every printed decimal, and the deliberately-wrong off-diagonal run reproduces the historical footgun row (10.674/2.081/80.106/0.710) exactly. Same box/env/seed/code-path ⇒ this is the expected signature of a faithful port (env-bridge precedent), and any conditioning-path deviation would have shattered it.
2. **Every load-bearing line is mutation-tested.** ~30 mutation checks across the rounds (rotation sign, orbit divisor, angle values, dispatcher bypass, guard removal, EMA hybrid, suffix truncation, …), each proven red then restored byte-exact.
3. **Cross-model review** at xhigh caught real issues at every round (13 findings total incl. 1 BLOCKING guard-authority spoof), all fixed and re-verified.
4. **Honest scope:** the port is the *as-run* B-F method (loop orbit, no `max_fwd_samples` chunking, no fa_cartesian); additions beyond as-run are all fail-closed safety/provenance features (trained-as guard, shared validation, collision-free naming, exclusive EMA loads) — none changes numerics, as the bit-identity proves.

## Caveats
- Acceptance is the D12 two-cell identity check, not the K={1,8}×5-seed matrix; the 5-seed registered rows remain the statistical reference.
- The smoke proves 3 optimizer steps, not a full 67.5k-step training; step-time and long-horizon stability are inherited claims from the source lineage.
- `generation.py`'s raw-conditioner path is documented, not dispatch-routed (D11).

## Next steps
1. Yixun: anonymous weights URL (→ neutralize `download_weights.sh`) and the `.git`-metadata decision for the anonymous share.
2. Optional: 5-seed acceptance matrix if a registered ORBITRIR row is wanted; pin the 4 sibling configs' `revision` before any variant experiment.
3. OrbitRIR-specific work (C8/C16/C32 orbits) starts from `frame_avg_angles` — config-only, per the Cn analysis in this folder's worklog.
