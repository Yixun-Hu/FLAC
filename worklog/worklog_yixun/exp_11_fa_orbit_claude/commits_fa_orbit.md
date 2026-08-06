# Commits — exp_11_fa_orbit

Base: `b9e38ce` (exp_10 model_comparison rows). Branch: `check-equivariance-necessity`.

| Order | SHA | Summary |
|---|---|---|
| 1 | `f8d18d3` | scaffold + plan Rev 2 (post-Codex REJECT plan review, 14 findings addressed) |
| 2 | `b1c1198` | round 1 — arm configs `FLAC_AR_BF_C{4L,8,16,32}.json` (orbit + no-grad-ckpt) + TDD tests `test_exp11_orbit_configs.py` and the Cn parametrizations in `test_invariant_conditioning.py` |
| 3 | `91cfc0e` | round 1 fixes (Codex REJECT b1c1198: 2 BLOCKING + 2 NIT) — strict `is True`/`is False` gc-leaf assertions + falsy-`0` regression test, averaging test parametrized over C8/C16/C32, duplicate-key/NaN-rejecting JSON loader, orbit tests 175 s → 14 s |
| 4 | `43a4d5b` | plan Rev 3 (approved fast-recipe amendment: no ViT grad-ckpt, micro×N=64 rungs, P0 profiling stage) + round-1 review-loop artifacts |
| 5 | `e566513` | round 2 — P0 profiling kit: `p0_profile.sbatch` (paired 10/30-step cell, mem_probe-derived gates, UUID-bound VRAM poll, `P0RESULT` line), `p0_submit_matrix.sh` (rung × orbit matrix + `spot` cells), `p0_collect.py` + TDD `test_exp11_p0_collect.py` (pairing, steps/s, attribution columns, `p0_report.md`) |
| 6 | `ec0250d` | round 2 fixes (Codex REJECT e566513: 7 BLOCKING + 2 NIT) — torchrun DDP + world-size/rank-placement gates (PL would have run world size 1 under `--ntasks=1`), in-fit step-10/30 timing via new `p0_runner.py` (single job per cell, no wall-time pairing), manifest-bound collection (runid/sha/jobid/config_sha, every expected cell reported, derived tables withheld + nonzero exit unless all-OK), cell-derived config map with a semantic orbit/grad-ckpt gate, sequenced util/power poller with per-tick completeness + stop-file lifecycle, exact-{VAN,C4L,C8} fit gating with AMBIGUOUS detection, worker-pair mode |

| 7 | `60764c1` | round 2 fixes v2 (Codex re-review of ec0250d: 6 BLOCKING + 2 NIT) — **FA1 attribution control** `FLAC_AR_BF_FA1.json` (single-angle orbit: same cylindrical pose path as C4L/C8, one ViT pass) with the fit moved to the exact {FA1, C4L, C8} set and VAN demoted to a separate vanilla contrast; manifest `mode` + exact provenance binding (maxsteps/mb/ngpu/workers, ns+random run id, no-clobber publish, run-specific report path); poller evidence mandatory (finite util/power per tick, `pollcsv_sha256` hash-verified, in-window ticks per UUID); per-cell Slurm time limits for the C16/C32 spots; worker pair in ONE manifest keyed `(cell, workers)`; MAXSTEPS pinned to exactly 30; config-map hook honoured only outside a job |

| 8 | `8d53691` | round-2 review-loop artifacts (P0 kit reviews + worklog closure) — main session |
| 9 | `aa4bc18` | P0 smokes live-validate the kit; C4L_32x2 no-ckpt OOM (real bound), FA1_32x2 1.01 steps/s — main session |
| 10 | `7e617a2` | P0 matrix launched (13 cells, jobs 3638637-49) + command record — main session |
| 11 | `72a8114` | round 3 — arm training launcher `fa_orbit_train.sbatch` (torchrun rung-parameterized 16x4/8x8, 32x2 barred by the measured OOM; arm→config map + semantic orbit gate; commit/drift, wandb, VRAM and world-size gates incl. a 300 s early-abort watchdog; exp_10-style INITIAL/RESTART lineage; dual-tee + atomic manifest; DRYRUN argv-parity vs exp_07), `assert_arm_configs_exp11.py` (exp_07's DINOv3-pin + init-identity checks re-pointed at the four arms), `fa_orbit_train_guardtests.sh` (33 cases, all green) |

| 12 | `5557974` | consolidated round — grad-ckpt pivot for all arms (no-ckpt OOM-infeasible for C8+), C4L byte-identical to exp_07 B-F, P0 poller `%.6g` precision fix, launcher round-3 fixes (pins, deep restart preflight, run lock, tested exit taxonomy, env/W&B gates, SMOKE mode, 52-case guard suite) |
| 13 | `0df4103`, `4e84485` | Q3/Q4 records, plan Rev 4 note, C32 authorization — main session |
| 14 | `abbff5a` | NEW-1a — `FLAC_AR_VANCKPT.json` (canonical + the two grad-ckpt leaves) + deep-diff test: the canonical manifest could never pass the post-pivot gate |
| 15 | `c89d05a` | NEW-1b/NEW-2/NEW-5 — P0 kit pivoted (VAN→VANCKPT, CKPT4 retired everywhere, matrix now 12 cells), OUTPUT_ROOT literal under Slurm, poller comment corrected and the 2 s liveness bound restored |
| 16 | `983a7ff` | launcher residuals — flock ownership (B3), fail-closed manifest-commit binding (B2), pip-freeze/final-tee/preflight-transcript durability (B5), exported W&B entity + post-run run-identity verification (B7), intent manifest before sbatch (NEW-3), safe FIFO (NEW-4) |
| 17 | `_(this commit)_` | fresh evidence — 72-case guard log and 78-case pytest log committed; ledger updated |

Notes: a commit cannot contain its own SHA (exp_07's amend lesson — never amend a SHA
into the commit that carries it), so each round's hash is reported with its output and
backfilled by the next exp_11 commit (row 7 backfilled here).
