# Commits — exp_07_fa_scratch

Base: `0bd5da0`. Branch: `check-equivariance-necessity`.

| Order | SHA | Summary |
|---|---|---|
| 1 | `b2e2000` | scaffold — from-scratch fa_invariant (Route B) |
| 2 | `79b9791` | plan — two matched from-scratch arms; budget decision table for Yixun |
| 3 | `8db486a` | plan review (REQUEST-CHANGES) + revision — 67.5k-step anchor discovered in FLAC.ckpt; --max-steps round; eval protocol; thresholds |
| 4 | `8ae9837` | config-identity audit round (Yixun's go-condition): released-ckpt probe v2 (counters/optimizer/scheduler/config-diff/ViT pin), arm configs BV/BF + asserts v4 (init-identity + fail-closed pin gate, red/green-proven), audit doc + launch manifest, plan eff-64 corrections, gpt-5.6-sol review→reverify→reverify2 loop closed; SOP reviewer-model update rides along |

Notes: audit authored by Fable 5 (main session; seat restored from Opus 4.8 per Yixun's `/model` switch). Reviewer for this and all future rounds: Codex `gpt-5.6-sol` xhigh (per Yixun 2026-07-10; CLI 0.144.1).
| 5 | _(this commit)_ | bookkeeping — record audit-round SHA `8ae9837` (amend during commit 4 changed its hash; lesson: never amend a SHA into a commit that contains it) |
