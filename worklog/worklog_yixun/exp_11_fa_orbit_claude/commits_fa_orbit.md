# Commits — exp_11_fa_orbit

Base: `b9e38ce` (exp_10 model_comparison rows). Branch: `check-equivariance-necessity`.

| Order | SHA | Summary |
|---|---|---|
| 1 | `f8d18d3` | scaffold + plan Rev 2 (post-Codex REJECT plan review, 14 findings addressed) |
| 2 | `b1c1198` | round 1 — arm configs `FLAC_AR_BF_C{4L,8,16,32}.json` (orbit + no-grad-ckpt) + TDD tests `test_exp11_orbit_configs.py` and the Cn parametrizations in `test_invariant_conditioning.py` |
| 3 | `_(this commit)_` | round 1 fixes (Codex REJECT b1c1198: 2 BLOCKING + 2 NIT) — strict `is True`/`is False` gc-leaf assertions + falsy-`0` regression test, averaging test parametrized over C8/C16/C32, duplicate-key/NaN-rejecting JSON loader, orbit tests 175 s → 14 s |

Notes: a commit cannot contain its own SHA (exp_07's amend lesson — never amend a SHA
into the commit that carries it), so each round's hash is reported with its output and
backfilled by the next exp_11 commit (row 2 backfilled here).
