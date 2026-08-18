# exp_18 loc_invert — lab notebook (append-only)

## 2026-08-18T15:05:00-0400 — Branch + scaffold

- **Goal** — Start the localization preflight per Yixun's query: new branch off `check-equivariance-necessity`, SOP scaffold for exp_18.
- **Change** — `git checkout -b localization-exp` (from `check-equivariance-necessity` tip `6170007`, clean tree). Created `worklog/worklog_yixun/exp_18_loc_invert_claude/` with `loc_invert_yixun_query.md` + this notebook.
- **Version Control** — branch `localization-exp`, base_commit `6170007`. Commits on this NEW branch cannot trip the cluster's content-scoped `EXPECT_SHA` gates (those bind to SHAs of `check-equivariance-necessity` checked out by pending legs; this branch is never checked out there). SOP directory conventions already match this repo (`worklog/worklog_yixun/`); no SOP edit needed.
- **Result** — passed (scaffold in place).
- **Next** — Recon for the plan; then `plan_loc_invert.md`; then Codex plan review; then Yixun approval gate.

## 2026-08-18T15:20:00-0400 — Recon findings that shape the plan

- **Goal** — Verify, before planning, every repo/data/API fact the protocol depends on.
- **Result** — passed; findings:
  1. **Announcements 01–06 read.** Binding here: full existing eval configs only (unseen = 6337 items / 17 rooms, verified again from `data/AR/unseen_eval.json`); TDD in `src/tests/`; eval-protocol flags (`--cond-method` etc.) declared in every manifest; universal Codex review.
  2. **Context construction excludes the target source by design** — `AR_md.get_ir_and_location_for_other_sources` draws references only from OTHER source nodes at the SAME receiver (`np.random.choice` over remaining nodes). So "h_obs excluded from context" and "all RIRs sharing the target source excluded" hold automatically for every query; no dataset-config change needed.
  3. **Latent release-code quirk**: metadata/context path building uses `"S00"+str(node)` — correct only for single-digit nodes ("S0010" ≠ "S010"). Seen split contains `S010_*` files. Consequence: (a) candidate-set code must parse node ids numerically and search metadata filenames robustly; (b) `SampleDataset.__getitem__` silently substitutes a random item on any load failure — the driver must verify per-item identity (`sample_target_id`) and log/flag substitutions instead of silently accepting them. Verify actual metadata naming at the data-readback rung once the dataset is on this box.
  4. **Reuse surface in fork `eval_FLAC.py`**: `evaluate_model` model-loading/EMA-remap path, cond-method dispatch (`--cond-method {vanilla,fa_invariant}`, `--cond-autocast`), `sample_context_ids` (6-decimal context fingerprint), `source_sha`, `orbit_provenance`, `build_metrics_record` style. Localization driver reuses these instead of reimplementing, so FA/yaw-aug/cyl arms run later under their correct protocols (announcement 05).
  5. **AGREE scorer API confirmed**: `src/metrics/metric_callback.loading_AGREE_model(ckpt, device)` → CLIP-style model; `model.encode_audio(wav, normalize=True)`; `Retrieval.compute_audio_features` pads/crops to 10240 samples first. Available checkpoints (HF `AmandineBtto/AGREE`): `AGREE_fullAR.pt` (saw FULL AR incl. unseen rooms — eval-only, leaky for this use), `AGREE_AR.pt` (train-split only — clean scorer for unseen-room claims).
  6. **This box (mae-cab-lab-server, 2×A6000, both idle) currently has NO dataset and NO model weights**: no `AcousticRooms/` (symlink or dir), `weights/` holds only exp_01 metric JSONs, `outputs_FLAC/*` hold no `.ckpt` (dirs re-created today 14:36 — an rsync may be staged). Yixun confirms checkpoints not yet rsynced. Preflight = plan + reviewed code + unit tests now; smoke/full runs once data+weights land. Released `FLAC_EMA.ckpt` + AGREE ckpts are downloadable from HF via `download_weights.sh` independent of the rsync.
  7. **Compute basis** (exp_01 log, this box): full 6337-item unseen eval, batch 64, ≈3.9 s/it × 100 it ≈ **6.5 min per seed**. Localization at M≈10 candidates × K=8 samples ≈ 10× conditioner cost + 80× (cheap) 1-step DiT+VAE per query ⇒ projected ~1–4 h for the full unseen sweep on one A6000; a timing probe rung will measure before launch.
  8. **Candidate sets are discrete and small**: AR rooms have ~10 source nodes (e.g. 250-item rooms = 10 sources × 25 receivers) ⇒ M≈10, GT included per spec. Random-candidate baseline is exactly computable (uniform over C per query), no Monte Carlo needed.
  9. **Oracle subtlety**: for candidate = GT source at the same receiver, the "GT candidate RIR" IS h_obs ⇒ cosine 1 ⇒ identity-oracle is trivially 100% and only a pipeline sanity check. A non-trivial upper bound needs a second measurement of the same pair (e.g. another `single_channel_ir_*` channel folder if the dataset provides one) — to verify at data readback; plan carries both variants conditionally.
- **Analysis** — No blocker to writing the plan; the run phase is blocked only on data/weights arrival. Protocol-integrity risks identified early: silent-substitution, S010 naming, scorer leakage, identity-oracle triviality, post-hoc τ/K tuning (solved by dev-split tuning + offline re-aggregation).
- **Next** — Write `plan_loc_invert.md`; Codex plan review; revise; surface open decisions + plan to Yixun for approval.

## 2026-08-18T15:25:00-0400 — Codex reviewer UNAVAILABLE on this box (401); declared fallback engaged

- **Goal** — Run the SOP-mandated plan review (Codex gpt-5.6-sol, xhigh).
- **Command / Validation** — `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message … < /dev/null`; prompt archived in session scratchpad.
- **Result** — failed: `401 Unauthorized — Missing bearer or basic authentication` on every transport; `~/.codex/auth.json` absent ⇒ Codex has never been authenticated on this box (mae-cab-lab-server). Infra classification: **infrastructure (auth), not a bug**.
- **Analysis** — Per SOP role table + CLAUDE.md, the declared fallback reviewer when Codex is unavailable is **Claude Opus 5 at max effort**, with the substitution stated in the review by-line (never silent). Fallback engaged for the plan review round; Yixun asked to `codex login` on this box so subsequent rounds (code reviews) can return to Codex.
- **Next** — Opus 5 fallback plan review → address findings → revise plan → surface to Yixun with open decisions.

## 2026-08-18T15:38:00-0400 — Codex re-authenticated by Yixun; review of record launched

- **Goal** — Return the plan review to the SOP-preferred reviewer now that Yixun ran `codex login` on this box.
- **Command / Validation** — same read-only xhigh `codex exec` invocation as the 401 attempt (prompt unchanged, archived in scratchpad).
- **Result** — launched. The in-flight Opus 5 fallback review is RETAINED as a supplementary independent opinion (`loc_invert_opus_plan_review.md`); the Codex review is the review of record (`loc_invert_codex_plan_review.md`). Both sets of findings will be addressed before the plan goes to Yixun.
- **Next** — Consolidate findings from both reviews → revise plan → approval package to Yixun.

## 2026-08-18T15:35:52-0400 — Codex RESTORED on this box; reviewer role returns to Codex

- **Goal** — Verify Codex after Yixun ran `codex login` on mae-cab-lab-server (his direct request this turn).
- **Command / Validation** — `~/.codex/auth.json` present (mode 600, created 15:25); `codex-cli 0.147.0`; end-to-end ping with the exact SOP invocation shape: `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message <scratch>/codex_ping.txt "<no-install connectivity prompt>" < /dev/null`.
- **Result** — passed: banner confirms `model: gpt-5.6-sol / sandbox: read-only / reasoning effort: xhigh`; reply `CODEX-OK gpt-5.6-sol` both on stdout and in the `--output-last-message` file; 2,991 tokens; session id `01a0165e-e5e1-7460-b749-c03f095448f8`.
- **Analysis** — The 15:25 entry's infra blocker is CLEARED. The Opus 5 fallback declared there was engaged but its review round had NOT yet run, so **no fallback by-line is owed anywhere**; the plan review (and all subsequent code reviews) run on the primary reviewer, Codex gpt-5.6-sol @ xhigh. Standing rule unchanged: every review prompt explicitly forbids installing/modifying environments (`-s read-only` is not an environment guarantee, issue_report §10).
- **Next** — Two open decisions surfaced to Yixun before the plan-review round fires: (1) does RAF enter exp_18 scope (future real-data localization arm noted in the plan) or was his RAF question standalone; (2) "go" on the Codex plan review of `plan_loc_invert.md`.

## 2026-08-18T16:05:00-0400 — Codex plan review received (REQUEST-CHANGES) → plan Rev 2

- **Goal** — Review-of-record on plan Rev 1 (`4f3658e`).
- **Result** — Codex `gpt-5.6-sol` xhigh (served model confirmed from run log): **REQUEST-CHANGES**, 2 BLOCKER / 7 HIGH / 1 MEDIUM / 1 NIT. Saved as `loc_invert_codex_plan_review.md`.
- **Analysis — disposition (ALL 11 adopted, none contested):**
  - C1 BLOCKER (context poses reveal target by exclusion → uniform baseline uninformative): adopted — context-conditioned exact baseline is now the REGISTERED comparison target; eligible-set sizes + context-member-prediction rate reported (§1, §2.6).
  - C2 BLOCKER (silent substitutions must fail closed, not be excluded): adopted — pre-generation identity audit, split hash, exact-6337/17 gate (§2.1).
  - C3 HIGH (estimand inconsistency; scene≠room): adopted — primary = pooled median over 6,337; `room_id = scene/scene_id`; macro = labelled secondary; per-seed ± SD separate from 17-room clustered bootstrap CI (§2.6).
  - C4 HIGH (dev selection could replace the specified method): adopted — LME/K=8 fixed; τ-only selection, pre-registered grid, smallest-τ tie-break (§2.5).
  - C5 HIGH (announcement 05 completeness): adopted — `vanilla / --rotate-deg 0 / autocast default / fa-angles n/a` pinned; fail-closed on nonzero rotate-deg (§2.3).
  - C6 HIGH (AGREE preprocessing claim false): adopted after **independent code verification** (`metric_callback.py:114` max_len=8000 for AR; `:287-288` truncation; `:312` retrieval.update; Retrieval pads back to 10,240) — registered scorer = first-8000 path + embedding-equality test vs the real route (§2.4).
  - C7 HIGH (candidates from metadata, not filenames): adopted — `enumerate_metadata_sources` is the candidate authority; file cross-check at readback; oracle eligibility separate (§2.2, §4.2).
  - C8 HIGH (parity/TDD insufficient for a second driver): adopted — explicit import boundary, one-query numerical parity test (identical waveforms), fail-closed on ARE/non-RF ckpts, full per-function contracts for driver + heatmap script (§4.5, §4.6).
  - C9 HIGH (compute claim wrong: 80>64; 1–4 h unsupported): adopted — R0 = fit+timing probe, rung 6 not waived, honest 8.7–13.7 h bound until measured, 2-GPU strategy = one seed per GPU, sharding only with a disjoint-union merge gate (§5, §6, §9).
  - C10 MEDIUM (noise not execution-stable): adopted — noise bank keyed (seed, query_id, k), shared across candidates (common random numbers), permutation/batch-split equivalence tests (§2.3, §4.3, §4.5).
  - C11 NIT (viz overclaims): adopted — candidate-extent labelling, T_disp = registered τ, pre-registered smoke query ids (§2.7, §5).
- **Change** — `plan_loc_invert.md` rewritten as Rev 2 (Rev 1 preserved at `4f3658e`).
- **Next** — Fold the supplementary Opus fallback review when it lands (delta pass); then approval package to Yixun.
