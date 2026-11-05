> **STATUS: DRAFT r0 (2026-11-04 ≈ 23:45 EST) — awaiting Codex plan review and Yixun's decisions on the flagged deviations (§7) and open questions; NOTHING runs on a GPU before both.** Drafted by an Opus 5.5 research subagent for the Planner; evidence in `orbitrir_yaw_evidence_research_notes.md`.

# plan_orbitrir_yaw_evidence — exp_35 (DRAFT for the Planner; not yet reviewed by Codex or approved by Yixun)

Draft by Claude Opus 5.5 (research subagent), 2026-11-04 23:30 EST. Evidence: `research_notes.md` (same folder).

## 1. Goal and what is reported
Give OrbitRIR (B-F, `exp07_BF@40k`, NAS `ar_40k_endpoints/BF`, sha `5319feb4…`, `--cond-method fa_invariant`) the
yaw-invariance evidence CANON has in exp_25, then state CANON vs OrbitRIR on both axes (θ = 0 quality, invariance).
No training. Deliverable `exp_35_orbitrir_yaw_evidence_claude/orbitrir_yaw_results.{md,json}`:
(a) ONE θ = 0 table: FLAC (P1-pin, exp_25), CANON (exp_25), OrbitRIR (B-F T, exp_35) — T60, C50, EDT, R@1/5/10,
K = 1 and 8, mean ± sd over seeds 42–46, item-weighted, unseen AR (6,337); the historical B-F row printed beside it.
(b) the same six metrics at 45° and under random yaw for the three models.
(c) the Tier 3 spread table (Δ_C4, 45° probe, Δ_R, sd(m_Z), item-weighted, exactly `depth_canon_collect.py`'s
estimators) with a B-F row next to CANON / P1-pin / YAW-pin (those three read from the committed exp_25 results JSON),
plus B-F's Tier 2 (dumps) and T-dup; and the off-orbit figure `analysis/yaw_robustness/generated/yaw_offorbit_metrics.{png,pdf}`
+ `yaw_offorbit_metrics_summary.csv` (0/45/135/225°, FLAC / OrbitRIR / CANON, K = 1 and 8, 5 seeds).

Pre-registered expectations (written before any cell): E1 the P1 control cells reproduce exp_25's P1-pin T s42
(bit-exact dumps under the no-flash shadow; else inside the vanilla envelope of research_notes (d)); E2 B-F θ = 0 s42 vs
exp_10 inside |ΔT60| ≤ 0.001, |ΔC50| ≤ 0.0005, |ΔEDT| ≤ 0.005, |ΔFD| ≤ 0.0005, |ΔR@k| ≤ 0.2 pp → exp_10 B-F cells
admissible for (b)/figure (not for the registered estimators); E3 B-F C4 pairs at the numerical floor (Tier 2 C4 PASS
under a B-F control), Δ_C4 ≤ 10 % of P1-pin on the acoustic metrics (retrieval within 1–2 item flips of the threshold);
E4 B-F 45° probe and Δ_R ≫ sd(m_Z) → B-F fails Tier 3's R rule and Tier 2 at 45°/R by construction (C4-only invariance);
E5 B-F(135°) ≈ B-F(225°) ≈ B-F(45°) (C4 equivalence), CANON(135/225) ≈ CANON(0°), P1 degrades; T-dup exactly 0.

## 2. Reuse vs new cells (evaluator = exp_25 runner, `--expected-stream-count 6337`, batch 64, EMA, cfg 1, 1 step, bf16 cond)
Reused, never re-run: exp_25 CANON / P1-pin / YAW-pin results at `d6ca307a` (Tier 1–3, θ = 0, 45° s42, R, T-dup);
exp_10 A6/a6c4 per-seed values for P1 and B-F at 0/45/90/180/270 (CSV only — the rotated raws are lost).
New (draft rows in `cells_draft.json`, exact exp_25 schema, rows 230–329 appended after TRAJ; names `exp35_*`):

| phase | block | cells | purpose |
|---|---|---|---|
| 1a | CTL (P1 T s42, K1/K8) | 2 | evaluator-identity control vs exp_25 P1-pin T s42 (re-run as a control, not a result) |
| 1a | BF-pin T s42 (K1/K8) | 2 | Yixun's "2 measurement cells" (B-F θ = 0 vs exp_10) |
| 1b | BF-pin T s43–46 | 8 | Δ_R partners, sd(m_Z), Tier 2 θ = 0 references (exp_10 θ = 0 has no stream/dump) |
| 1b | BF-pin R s42–46 | 10 | random yaw |
| 1b | BF-dup | 2 | T-dup determinism (K1 + K8, as CANON) |
| 1b | BF-pin C4+45 s42 | 8 | one-pin Δ_C4 / 45° probe + Tier 2 C4 pairs (recommended; see D2) |
| 2 | OFF-P1, OFF-BF (135/225) | 40 | figure |
| 3 | OFF-CANON (135/225) | 20 | figure |
| 3 | CANON-45 (45°, s43–46) | 8 | 5-seed CANON at 45° (exp_25 has s42 only) — optional |
Total 100 (Yixun's literal list was 11 + 60).

## 3. Evaluator choice
A new pin is unavoidable (the runner checks `make_cells.py --check` and a strict, committed HEAD). Code evidence that the
eval path equals `d6ca307a`'s: `git diff d6ca307a 7bff64b8 -- eval_FLAC.py src ':(exclude)src/tests' data/AR` touches only
6 `src/localization/*` files, imported by nothing outside `src/localization` and the tests; the arm configs, the runner,
collector, generator, cells and probe are byte-identical in the main checkout and the exp-25 worktree. **But the numerics
backend changed:** `flash-attn 2.7.4.post1` was installed into `flac` on Nov 1 (pip-freeze `e896c48f…` → `8b7a802e…`);
exp_25 ran without it, exp_07/exp_10 with it. Recommended: run exp_35 with the import shadowed (`PYTHONPATH=<scratch dir
holding flash_attn/__init__.py that raises ImportError>` exported to the runner; no env or code change; every cell log then
prints the shadow's message), and require the 2 CTL cells to reproduce exp_25's P1-pin T s42 dumps bit-exactly (waveform
sha256 from the force-added exp_25 receipts) — that proves "eval path unchanged". Alternative: run with flash and accept
CTL within the vanilla envelope (disclosed). Either way: run from a dedicated worktree (new branch `exp-35-orbitrir-yaw`
from `localization-exp` HEAD; CANON@40k / P1 / BF checkpoints staged into its cache, +2.2 GB) — never from the main
checkout, whose HEAD moves with records and exp_22 work (runner aborts with PIN DRIFT, exit 4).

## 4. Per-file planned changes (TDD, < 200 lines per commit, Codex review per round)
- `exp_25…/depth_canon_make_cells.py`: arm `BF` (existing arm schema), per-arm flags (BF: `--cond-method fa_invariant
  --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64` + common flags) and per-arm `cond_method` for the
  `build_output_paths` suffix (`_fa_invariant_a4`); `exp35_` prefix; blocks CTL, BF-pin, BF-dup, OFF-P1, OFF-BF,
  OFF-CANON, CANON-45 appended (priorities 10–16). Tests: the first 230 rows byte-identical to the committed file;
  counts/order; BF argv + suffix; names/suffixes injective incl. vs exp_25; `--check` round trip.
- `depth_canon_cells.json`: regenerated (330 rows).
- `depth_canon_collect.py`: `ARM_PROTOCOL` (vanilla default; BF = fa_invariant / [0.0, 90.0, 180.0, 270.0] / "batched" /
  64) used by `record_reasons` (:123-125) and `cell_paths` (:455-458); `make_pairs` / `_index` admit CTL, BF-pin,
  BF-dup (dup excluded from `_index` like T-dup); `tier2` / `tier3` / `table_block` / `render_md` / `ARM_LABEL` gain BF;
  an exp_35 collect mode = exp_35 blocks only at `--pin <H35>` plus read-only inputs: the exp_25 results JSON (sha256-pinned,
  pin `d6ca307a`) for CANON / P1 / YAW rows and P1-pin Δ_C4, and the force-added exp_25 T receipts/streams for the CTL
  bit-exact check and the cross-campaign `input_hash` identity (B-F T/R must reproduce exp_25's hashes per K/seed);
  outputs to the exp_35 folder. exp_25 behaviour unchanged (existing tests stay green, incl. `gen_model_comparison`'s
  exp25 contract).
- `depth_canon_cond_probe.py` (only if Tier-2 option A): `--cond-method fa_invariant` (A = fa(md), B = fa(rot_d(md)), no
  canonicalization), registered contract bound to the BF sha and H35; control used at d128/d256/d384 only.
- `analysis/yaw_robustness/plot_yaw_curves.py`: `--mode offorbit` (angles 0/45/135/225; sources: θ = 0 and 135/225 from
  metric JSONs, FLAC/OrbitRIR 45° from the committed `yaw_metrics_raw.csv`, CANON 45° from exp_25 s42 + CANON-45;
  third style for CANON; labels FLAC / OrbitRIR / CANON; per-file checks of angle, seed, cond_method, n 6337, pin;
  writes the raw + summary CSV and png/pdf). Existing modes untouched. Test file `src/tests/test_plot_yaw_offorbit.py`.

## 5. GPU plan, monitoring, time
One lane: `EXPECT_HEAD=<H35> ALLOW_SHARED_GPU=1 [PYTHONPATH=<shadow>] bash …/depth_canon_eval_run.sh --lanes 1 --only …`
in chunks of ≤ 2 cells (`--only` / growing `--limit`; complete cells are skipped), never signalling anything.
Headroom: GPU 1 has 40,080 MiB free beside Yixun's 8,248 MiB rank; exp_25 lanes used 1.9 GiB (B-F expected 2–4 GiB,
measured in 1a); runner floor 10 GiB; his queue's own eval gate (19,160 MiB) stays satisfied.
**Criterion (his log, read-only tail of `train_exp30_ACEattn_s43_*.log`, s/step from the tqdm elapsed field):** baseline
4.22–4.24 s/step; during his own table job 7.61–9.71 per 500-step window, mean 8.52 (×2.01). While our lane runs: every
completed 500-step window ≤ 9.71 s/step AND the mean since our first spawn ≤ 8.52; checked after every chunk; violation
→ no further spawn, report to Yixun. When his training ends, move to `--lanes 0` (GPU 0 free while his s43 table job runs
on GPU 1).
Per-cell planning values (co-tenant; from exp_25 medians and exp_10 fa/vanilla ratios; replaced by 1a's measurements):
vanilla/CANON K1 17.6, K8 19.2 min; B-F K1 ≈ 24 (18.5–28), K8 ≈ 35 (29–41) min; free card: vanilla ≈ 14.5 / 15.7,
B-F ≈ 19 / 21 (uncertain). Phase 1a ≈ 1.6 h (+ ≈ 5 min probe); 1b ≈ 13.8 h (11–16) → **(i) ≈ 15.4 h (12.4–17.9)**,
minimal variant (no C4+45, 1 dup) ≈ 11 h. Phase 2 ≈ 16.0 h, phase 3 ≈ 6.1 h (+2.5 h CANON-45) co-tenant →
**figure cells ≈ 22–25 h on one co-tenant lane** (≈ 17–19 h on one free card, ≈ 8.5–9.5 h on two) → report to Yixun
before phase 2 (≈ 1 day rule). Total ≈ 40 lane-hours co-tenant. Impact: ≈ 0.5 h of his training per co-tenant hour
(his table job cost ≈ 4.9 h; phase (i) alone ≈ 7.7 h; all phases co-tenant ≈ 15–20 h). His s43 ETA unshared ≈ Nov 6
10:10 EST. Collector ≈ 15 min CPU; NAS dumps ≈ 26 GB.

## 6. Risks
Retrieval Tier-3 thresholds for B-F (10 % of P1's Δ_C4 = 3.5–6.8 items) are of the order of numerical jitter; NAS/CIFS
stalls (dataset substitution → caught by `expected_stream_count` + cross-campaign `input_hash` checks); the co-tenant
slowdown may exceed ×2 for fa cells (criterion stops us); the exp_22 free-card watcher must not pre-empt the lane;
the shadow trick needs Yixun's consent; mixed provenance in the figure (45° FLAC/OrbitRIR from exp_10 CSV, flash era).

## 7. Deviations from Yixun's request (flagged)
D1 B-F T ×10 at the new pin are required (exp_10 θ = 0 has no stream/dump). D2 B-F C4+45 ×8 recommended (registered
estimators need one-pin input_hash pairs; the exp_10 C4 set mixes loop/batched evaluators and a neuronic K8 s42 cell).
D3 CANON's composition control is invalid for FA → B-F probe (code) or labelled alternatives. D4 flash-attn backend change →
shadow + 2 CTL cells. D5 CANON 45° s43–46 (8). D6 T-dup ×2. D7 worktree, not the main checkout. D8 co-tenancy scoped
to phase (i) recommended. D9 runner logs land in the exp_25 folder (runner unchanged), copied to exp_35 afterwards.
D10 a re-measured B-F row in `model_comparison.md` needs a generator contract change + neuronic regen (propose: results
file only).
