# Commits — exp_11_fa_orbit

Base: `b9e38ce` (exp_10 model_comparison rows). Branch: `check-equivariance-necessity`.

| Order | SHA | Summary |
|---|---|---|
| 1 | `f8d18d3` | scaffold + plan Rev 2 (post-Codex REJECT plan review, 14 findings addressed) |
| 2 | `b1c1198` | round 1 — arm configs `FLAC_AR_BF_C{4L,8,16,32}.json` (orbit + no-grad-ckpt) + TDD tests `test_exp11_orbit_configs.py` and the Cn parametrizations in `test_invariant_conditioning.py` |
| 3 | `91cfc0e` | round 1 fixes (Codex REJECT b1c1198: 2 BLOCKING + 2 NIT) — strict `is True`/`is False` gc-leaf assertions + falsy-`0` regression test, averaging test parametrized over C8/C16/C32, duplicate-key/NaN-rejecting JSON loader, orbit tests 175 s → 14 s |
| 4 | `43a4d5b` | plan Rev 3 (approved fast-recipe amendment: no ViT grad-ckpt, micro×N=64 rungs, P0 profiling stage) + round-1 review-loop artifacts |
| 5 | `_(this commit)_` | round 2 — P0 profiling kit: `p0_profile.sbatch` (paired 10/30-step cell, mem_probe-derived gates, UUID-bound VRAM poll, `P0RESULT` line), `p0_submit_matrix.sh` (rung × orbit matrix + `spot` cells), `p0_collect.py` + TDD `test_exp11_p0_collect.py` (pairing, steps/s, attribution columns, `p0_report.md`) |

Notes: a commit cannot contain its own SHA (exp_07's amend lesson — never amend a SHA
into the commit that carries it), so each round's hash is reported with its output and
backfilled by the next exp_11 commit (rows 3–4 backfilled here).
