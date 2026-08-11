# Lab notebook — exp_10_fa_scratch_resume

## 2026-08-01 — scaffold
- **Goal** — resume exp_07's B-F (fa_invariant from-scratch, SyncBN-64 DDP recipe) from `outputs_FLAC/exp07_BF/FLAC_exp07_BF/exp07_BF/checkpoints/epoch=8-step=40000.ckpt`; all screens under the fa protocol (`--cond-method fa_invariant`), correcting the exp_07 mismatch.
- **Version Control** — branch check-equivariance-necessity, base = exp_09 closure (`93449cb`).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → launch.

## 2026-08-01T14:55:18-04:00 — plan Rev 2 APPROVED by Yixun → implementation round 1 (bf_resume_launch.sh)
- Coder = Opus 5 max seat; guard-tests + Codex review before use; then 15-step probe → launch.

## 2026-08-01T15:15:02-04:00 — round 1 SHIP (26/26 guards) → probe PASS (all lineage asserts; lr 4.903e-5 analytic; 15 fa steps) → COMMITTED RUN LAUNCHED (40k→67.5k, wandb)
- Acceptance criteria (pre-launch): full-state restore @40000, fa path active, 27,500 steps to 67,500, ckpt/2500, no OOM/NaN; screens per 2,500 with `--cond-method fa_invariant`; hard aborts only.
- Monitors armed (ckpt arrivals + death guard). ETA ~2.3 d → screens → R1–R3 readouts.

## 2026-08-04T11:21:39-04:00 — DECOMPOSITION CELL (vanilla P1@40k under fa eval, 5-seed both K): ensembling confound REFUTED — the invariance effect is training-side
- P1@40k + fa-eval: K8 8.817/1.0009/42.283/R4.049 · K1 10.257/1.0824/45.436/R3.926 — test-time 4-view averaging DEGRADES a vanilla model (R@1 −1.12, EDT +1.63 at K8 vs its own protocol; the model never learned averaged conditioning).
- Same-protocol matched-step comparison: fa-TRAINED 8.202/0.9778/38.79/R5.39 beats vanilla+fa-eval on all 12 cells (T60 −0.62, EDT −3.49, R@1 +1.34 at K8) → **fa's advantage is TRAINING-side invariance, not inference ensembling; "invariance makes the difference" licensed at matched steps (one training seed; 40k point).**

## 2026-08-05T15:45:00-04:00 — STALL DISCOVERED: run dead at step 65,000/67,500 (found during exp_11 reconnaissance)

- **Goal** — routine cross-check of exp_10 state while scaffolding exp_11 (orbit-size sweep).
- **Result** — `partial` (run stalled short of the pre-registered 67.5k endpoint):
  - Last checkpoint `epoch=14-step=65000.ckpt` written **2026-08-05T09:33:22-04:00**; no `train.py` process alive; all 8 GPUs at 0 MiB / 0% at 15:27.
  - Screens on disk through S62500 (fa protocol, K8 s42); S65000 not yet screened.
  - **The tee'd train log is GONE** (`fa_scratch_resume_2026-08-01_*_exp10_BF_train.log` absent from the exp folder), and every `wandb/run-*` dir from before 2026-08-05 is gone, while ALL tracked files carry mtime 2026-08-04T23:17 — i.e. an untracked-file wipe / re-checkout of this working tree happened at ~23:17 on Aug 4 (a second session's smoke runs `flac_readme_smoke_neuronic` appear in wandb from 00:39 Aug 5 onward). Checkpoints under `outputs_FLAC/` survived.
- **Analysis** — **infrastructure, not a real bug**, on the available evidence: the training process survived the Aug-4 wipe (ckpts kept landing on cadence: 60k @19:10 Aug 4, 62.5k @02:22 Aug 5, 65k @09:33 Aug 5) and died between 09:33 and ~15:00 Aug 5 with its stdout/stderr log destroyed, so the proximate cause (kill, crash, OOM from cotenant pressure) is not diagnosable. Step cadence had also slowed to ~0.095 steps/s (vs ~0.14 planned), consistent with cotenancy load. The 65k checkpoint is a clean resume boundary; `bf_resume_launch.sh` RESTART mode (EXPECTED_STEP=65000, ckpt inside `outputs_FLAC/exp10_BF/`) covers exactly this case — ~7.3 h × 2 GPUs to reach 67,500.
- **Next** — restart decision deferred to Yixun (surfaced 2026-08-05 together with the exp_11 plan): the R1 primary readout (B-F@67.5k vs P1@67.5k, 5-seed both K) is blocked until the endpoint exists. Also flagged: wandb lineage for this run is broken (log + run dirs destroyed); the on-disk ckpt/metric JSONs are the surviving record.

## 2026-08-05T21:05:00-04:00 — RECONCILIATION (two machines, one experiment)

- The stall entry above describes the CLUSTER copy of exp_10 (8-GPU machine; post-Aug-4-wipe, log/wandb loss, dead at 65k). **The ORIGINAL run on the A6000 box (this session) never stalled: it completed 67,500 at 2026-08-05 ~18:5x EDT and the endpoint gate block ran there** — the entry below records those gates. Checkpoints/metrics of record live on the A6000 box under `outputs_FLAC/exp10_BF/`; the cluster restart decision is MOOT (no restart needed — the run is done). Cross-machine metric-JSON consolidation per the pending metrics_json proposal. **Source-of-record provenance:** host `CAB-Lab-Server-8` (2×A6000 box); endpoint ckpt `epoch=14-step=67500.ckpt` sha256 `7dae243c573e2a48…`; all gated JSONs generated on this host (no cross-machine JSON transfer has occurred).

## 2026-08-05T20:49:19-04:00 — ENDPOINT GATES: R3 PASS (exact C₄; 45° breaks) · candidate rule NO QUALIFIER · R1 endpoint = band-worst draw → tier **SHORT** by the fixed rule; window reading contradicts the endpoint (pre-registered R1b)
- **R1 (fixed endpoint, 5-seed):** BF@67.5k K8 10.0412±0.0118/1.0050±0.0012/42.1068±0.0373/R6.6656±0.0607 vs P1@67.5k 8.757/0.9753/36.962/6.154 — P1 better T60/C50/EDT (z +15…+79), **BF better R@1 at BOTH K (z +2.9/+3.2)**. K1 same shape. → **SHORT** per the bounded tiers.
- **Endpoint anomaly (honest note):** the 67.5k draw (T60 10.04) is far outside BF's own screen band (8.58–9.42 over 42.5k–65k; 62.5k read 8.582) — a band-worst endpoint, the same phenomenon as exp_07 B-V's 67.5k endpoint (band max). The fixed-rule verdict stands (that is what pre-registration means), but R1b (window, exploratory): BF's screens track P1's curve on T60/C50/EDT and lead on R@1 from 50k onward.
- **R2:** 0/8 (endpoint). **R3: PASS** — C₄ spreads ≤0.0315 (T60 0.0009/C50 0.0001/EDT 0.0011), 45° control breaks (T60 +2.04, R@1 −2.05). Equivariance exact at the endpoint.
- **No extension offer** (plan: requires VIABLE-or-better). Standing insight preserved: at matched steps ≤65k, fa-scratch ≈ vanilla on error metrics + ahead on retrieval (the 40k 5-seed 12/12 result + decomposition remain the paper-grade evidence); the endpoint draw does not retract those.
- Next: closing package (results/analysis/HTML/closure review) — tomorrow morning.

## 2026-08-05T20:56:36-04:00 — CLOSURE: results/analysis/HTML written; closure review launched
- Verdict SHORT (fixed endpoint) + split retrieval win + R3 exact; standing 40k/decomposition evidence unchanged. Closing package committed; integrative closure review in flight; exp_10 closes on its verdict.

## 2026-08-05T21:09:36-04:00 — closure review CLOSE-WITH-FIXES → all fixes applied → exp_10 CLOSED
- All findings applied verbatim: 12/14-not-12/12 (FD excepted, both K); retrieval-lead claim narrowed to gated points (57.5k/60k single-seed deficits acknowledged); decomposition narrowed to interaction-evidence (not clean causal attribution); R2/R3 relabeled contextual (no registered candidate; S45000 cadence gap disclosed); R3 "exact" scoped to conditioning-by-construction with per-metric spreads incl. retrieval; P1-K8 s42 selcurve substitution documented; two-machine provenance (hostname + endpoint sha) recorded; plan header/date/HTML metadata fixed. Verdict SHORT stands. **exp_10 CLOSED.**

## 2026-08-08T00:34:45-04:00 — POST-CLOSURE ADDENDUM BLOCK (Yixun blanket approval overnight 2026-08-08): pre-registered before evals run
- **A1 (62.5k confirm):** 5-seed × both K of `exp10_BF/.../step=62500` — POST-HOC selected as the band-typical best screen point (s42 T60 8.582, only sub-released T60 in the resumed leg); labeled EXPLORATORY in all tables (the registered candidate rule found no qualifier; this row does not alter the SHORT verdict).
- **A2 (matched-compute readout):** fa-scratch at ~1/3.5 of the vanilla step budget vs vanilla at full budget — closes exp_10's open estimand approximately, using existing ckpts only; step-time ratio 3.5× from the exp_07/10 logs; approximation disclosed (no new training).
- **A3 (pre-40k band diagnostic):** fa-eval of exp_07 B-F 30k–37.5k (in flight) → resolves whether 40k was band-typical or a best-draw; addendum to the closing analysis either way.

## 2026-08-08T01:15:18-04:00 — A3 diagnostic COMPLETE: the 40k point is a band-best SPIKE (mystery resolved; honesty addendum owed on the 40k comparison)
- Pre-40k fa-eval band (K8 s42): 30k 8.858/1.0592/39.469/R4.134 · 32.5k 9.022/1.0350/40.608/R4.308 · 35k 9.733/1.0347/41.784/R5.050 · 37.5k 9.176/1.0273/41.453/R5.870 — vs 40k 8.190/0.9804/38.811/R5.302. **40k sits 0.7–1.5 T60 below every neighbor on BOTH sides; pre- and post-resume bands are the same (≈8.6–9.7).** Nothing degraded after 40k; the reference was exceptional.
- **Consequences:** (1) the exp_10 "weird regression" question is closed — endpoint-luck + band structure, no mechanism needed; (2) the 40k "12/14 fa-vs-P1 win" compared fa's SPIKE to P1's typical draw — band-vs-band, fa TRACKS vanilla rather than leading (fa pre-40k T60 band 8.9–9.7 vs P1's ≈8.6–9.2 in the same region); the matched-step claim must be restated band-level (addendum with A1/A2 results); (3) exp_07's correction addendum conclusion ("on par with vanilla") SURVIVES — it claimed parity, not superiority — but its evidence point was the spike; the band data now supports parity more honestly than the spike did.

## 2026-08-10T23:07:34-04:00 — A4 PRE-REGISTERED (before any eval runs): fair best-checkpoint comparison inside a 40k budget
**Yixun's directive (2026-08-10):** screen vanilla at FA's 2.5k density and restrict the comparison to a 40,000-step budget, so "best checkpoint at matched steps" is decided on equal draws.

- **Grid (identical for both arms):** every 2,500 steps, 2,500 … 40,000 = **16 points per arm**. Both arms have all 16 checkpoints on disk. Existing coverage was asymmetric (FA 6, vanilla 4) — that asymmetry is exactly what biased the earlier best-of read toward FA, so it is being removed rather than argued around.
- **Arms (matched recipe, single-delta configs, seed 42):** FA = exp_07 B-F (`FLAC_AR_BF.json`, `--cond-method fa_invariant`); vanilla = exp_07 P1 (`FLAC_AR_BVp1.json`, default conditioning). Each under its own protocol per announcement 05.
- **Screen protocol (pinned):** EMA weights, K=8, seed 42, cfg 1.0, steps 1, full unseen split 6,337/17, per-scene mean.
- **Primary readout (the fair version of the requested claim):** per metric, each arm's best value over the identical 16-point grid; report the per-metric winner and the tally. Equal draws remove the density bias (previously 15 vs 9).
- **Secondary (deployable, not per-metric cherry-pick):** ONE composite-best checkpoint per arm = minimum mean rank across the six metrics over the grid (ties → later step); those two checkpoints then get a 5-seed confirm at BOTH K, which is the comparison a model card could carry.
- **Disclosures fixed in advance:** selection runs on single-seed screens (eval σ ≈0.01–0.03 on T60, far below band swings); only the two composite picks are 5-seed confirmed; FD reported where present; a 40k budget favours neither arm a priori but is well short of both arms' 67.5k budget, so the result speaks to the 40k operating point only.
- **Nothing about this addendum changes exp_10's registered SHORT verdict.**
