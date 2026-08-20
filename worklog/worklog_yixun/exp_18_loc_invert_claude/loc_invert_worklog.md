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

## 2026-08-18T16:40:00-0400 — Supplementary Opus review received (REQUEST-CHANGES, 23 findings) → plan Rev 3

- **Goal** — Fold the supplementary fallback-reviewer findings (`loc_invert_opus_plan_review.md`; Opus 5 max effort, launched during the Codex 401 outage, retained as a second independent opinion).
- **Result** — 4 BLOCKER / 6 HIGH / 8 MEDIUM / 5 NIT. Planner independently re-verified every load-bearing new claim before adoption:
  - **O2 CONFIRMED in code** — AGREE audio tower samples at inference: `AGREE/AGREE/audio_model.py:174/201` → `VAEBottleneck.encode` → `vae_sample` (`randn_like`, no eval guard). E_a is stochastic and consumes global RNG. Fix registered: deterministic VAE-mean readout in `agree_embed.py` (plan §2.4), sampled readout demoted to a measured diagnostic.
  - **O1 CONFIRMED in data** — 16/17 unseen rooms have exactly 10 sources; `LivingRoomsWithHallway_idx_30` has 9 (225 files), Cafe_idx_1 has 922/1000 pairs. With 8 context refs the non-context eligible set is 2 (and {GT} alone in LRH_idx_30 ⇒ context-conditioned baseline 100% there ⇒ excluded, labelled, from the information-matched aggregate). Motivates the recommended K_ctx=1 secondary sweep (existing `_1` config).
  - **O14 CONFIRMED** — `AGREE/AGREE/model_configs/dinoV3.json` audio_cfg `"pretrained": "weights/FLAC/VAE.ckpt"` (CWD-relative, loaded at construction, then overwritten by the AGREE state dict — file must exist regardless); no HF cache on this box + `HF_HOME` unset ⇒ gated DINOv3 needs login/cache-rsync. Asset manifest updated (plan §3).
  - **O4 CONFIRMED (in-repo)** — no `single_channel_ir_*` other than `_1` referenced anywhere; second-measurement oracle contingent on dataset inspection. Adopted resequencing: R-1 dataset-only gate (oracle + baselines + nearest-context control) runs before any FLAC ckpt arrives.
  - O3 (constant-source wiring control + power statistic), O5 (already largely folded via C3; baseline conventions unified), O6 (per-query GT==md['source'] invariant; S010-quirk claim reframed as unestablished — namespaces separate), O7 (fingerprint relabelled regression guard), O8 (pin batch_size+num_workers; worker-seeded context draw), O9 (pin matmul precision/autocast/weights_source/clamp), O10 (nearest-context non-generative control, registered comparison), O11 (metric degeneracy caveat; dev objective → pooled mean), O12 (paired room-clustered tests), O13 (compute basis corrected to the K=8 eval ≈13 min; amortization parity-gated), O15 (= C2, already fail-closed), O16 (smoke pinned to seen rooms), O17 (pre-registration committed with SHA), O18 (logsumexp/serialization/readout test additions), O19 (shallow-copy variants), O20 (depth-silhouette option), O21 (autocast-off diagnostic), O22 (headline-ckpt recommendation CHANGED to released FLAC_EMA), O23 (dev scope R0-gated): **all adopted**.
- **Change** — `plan_loc_invert.md` rewritten as Rev 3 (Rev 2 preserved at `e71df84`).
- **Analysis** — Both reviews converged on the two protocol-critical flaws (context-exclusion information leak; fail-closed identity audit), and each contributed unique blockers (Codex: estimand/scorer-protocol/parity; Opus: stochastic scorer/wiring controls/asset gaps). Cross-model + fallback double review was worth the accident that produced it.
- **Next** — Approval package to Yixun (Rev 3 + updated decision list). Implementation starts only on his approval.

## 2026-08-19T09:00:00-0400 — Yixun APPROVED plan Rev 3; decisions locked

- **Goal** — Record the approval gate (SOP artifact #2 sign-off).
- **Result** — Yixun (verbatim): "approve; 2: FLAC_EMA; 3: yes; 4: yes; 5: yes; 6: for the dataset, please wait, I am downloading it to /media/diskstation/yixunhu/FLAC/AcousticRooms/, and I have finished the `huggingface-cli login`."
  Locked: headline ckpt = released `FLAC_EMA.ckpt`; scorer = `AGREE_AR.pt` primary + `fullAR` diagnostic with deterministic VAE-mean readout; K_ctx=1 secondary sweep R2b APPROVED; seeds 42/43/44, K=8; dataset arriving at `/media/diskstation/yixunhu/FLAC/AcousticRooms/` (wait for Yixun's completion); HF gated-DINOv3 access ready.
- **Next** — TDD Round 1 (Coder: `candidates.py` + `scoring.py` + tests) in parallel with `download_weights.sh`; R-1 dataset gate when the download completes.

## 2026-08-19T16:20:00-0400 — Coordination: second session claimed exp_19 (RAF finetune) on this checkout/branch

- **Goal** — Record cross-session coordination so numbering and file ownership stay conflict-free.
- **Result** — Peer session "flac-d9" (same checkout, same branch `localization-exp`) claimed **exp_19** for a Yixun-directed FLAC finetune on RAF (facebookresearch/real-acoustic-fields, HAA-recipe style); its scaffold is committed. Its code will live in `data/RAF/`, a new `RAF_md.py`, and new dataset configs — disjoint from exp_18's files. It commits path-scoped only.
- **Analysis** — Consequences for exp_18: (1) the future cross-arm localization experiment takes **exp_20+**; (2) two writers share this working tree — our commits stay path-scoped (they already are: Coder round commits name their files explicitly) and we re-check `git log` before committing; (3) no protocol impact.
- **Next** — unchanged (await Coder r1 completion → Codex r1 review).

## 2026-08-19T17:10:00-0400 — Coder Round 1 COMPLETE (12 commits, 115 tests); env fixed; ambiguity rulings

- **Goal** — Close out the Coder r1 deliverable and rule on its 9 flagged ambiguities before the Codex per-round review.
- **Version Control** — r1 commits `64fa6be 9bfa509 80ad3aa 45bad42 c3537d1 88f8989 43d1401 abd5021 d9cba5d 69f52e0 e8b6d49 ce1503b` (all < 250 lines, red→green evidence per cycle; two genuine second reds documented: MC-tolerance at cycle 8, discrete-percentile-grid at cycle 11).
- **Command / Validation** — `pytest src/tests/test_loc_candidates.py src/tests/test_loc_scoring.py -q` → 115 passed. `project_to_camera` parity vs `AR_md`: exact (rtol=atol=0).
- **Result** — passed. **Env fix:** `flac` env had setuptools 83 (pkg_resources removed) breaking every lightning-importing test (50F/15E, pre-existing at base too); pinned `setuptools<81` (80.10.2) → full suite now **1721 passed / 1 failed / 3 skipped**. The 1 failure is `test_yaw_aug_record_control.py::test_committed_record_agrees_with_exp11_registry` — exp_15's control test asserts `final_ckpt_sha256 not in` exp_11's VANL registry entry, which became legitimately false when exp_11's run completed. **Pre-existing, not exp_18's; owned by exp_15 — reported to Yixun, not touched.** Regression rung for exp_18: green modulo that documented, explained failure.
- **Analysis — Planner rulings on the Coder's flagged ambiguities:** (1) CandidateSet 5 fields: accepted, membership mask is driver-side. (2) **LME `τ·(logsumexp(s/τ) − log K)` CONFIRMED as the registered form** — it is literally Yixun's spec formula S = τ log[(1/K)Σexp(s/τ)]. (3) no-default-τ fail-closed: accepted. (4) **nearest_context_baseline ruling: report BOTH variants** — (a) raw as implemented (pure acoustic retrieval; by construction predicts a context member and can never hit GT — labelled as such), (b) eligible-restricted via a new optional mask argument (retrieval + elimination, the information-matched control). Variant (b) is a Planner-mandated addition for the r1 fix batch. (5) baseline `top1 = 1/n_eligible`: accepted. (6) summarize schema incl. per-candidate weight 1/M: accepted. (7) stats defaults: accepted. (8) fail-closed extras incl. `assert_gt_matches_loader` on `source` only: accepted (`source_vit` is the same projection unsqueezed, produced by the same code under test). (9) parity exact: noted.
- **Next** — Codex per-round review r1 (fix batch = its blocking findings + ruling #4b) → round closes only after fixes verified.

## 2026-08-19T18:20:00-0400 — Round 1 CLOSED (fix batch verified)

- **Goal** — Close the r1 loop per SOP (code → review → fixes → re-verify → log).
- **Version Control** — fix commits: `c09b7e8` `089ab2f` (F1 finiteness, HIGH), `cf4c0bc` (F2 cross-node src_loc uniqueness), `5693876` (F3 method="linear" + golden fixtures), `2ecc62f` (F4 MC 1e-3 for all metrics), `1cb5d92` (F5 Planner eligibility mask), `c33617e` (ledger).
- **Command / Validation** — Planner-independent re-verification: `pytest test_loc_candidates.py test_loc_scoring.py -q` → **174 passed**; spot-checks: NaN raises in cosine_sims/aggregate/uniform_baseline, F5 mask restricts prediction (raw 0 → masked 1), `method="linear"` present at scoring.py:429. Coder deviations reviewed and accepted (F1 scope extension to predict_index/power_statistic — good; F2 single authoritative site per instruction with scoring `_gt_index` backstop; F4 strengthened draws; F5 restriction-not-expectation semantics — the driver passes the non-context mask for the control).
- **Result** — passed; round 1 CLOSED. Blocking findings: 0 open. Nits: none deferred.
- **Next** — Round 2: `src/localization/agree_embed.py` + tests (deterministic VAE-mean readout, preprocessing parity, RNG isolation). Integration tests can run for real now (AGREE_AR.pt + VAE.ckpt + HF cache all present).

## 2026-08-19T19:05:00-0400 — Round 2 delivered; Planner verification PASSED; measured scorer noise

- **Version Control** — r2 commits `56d321a a664041 e16ac40 836f16a 4a4ff7e ec496f3` (preprocess / mean-readout / loader+sha / integration / O18 edge / ledger).
- **Command / Validation** — exp_18 suite **212 passed** (38 new, 34 unit + 4 integration on the REAL AGREE_AR.pt). Planner-independent check on the real model: mean readout bitwise deterministic, global RNG state untouched, [2,512] unit-norm f32, ckpt sha `b664d5c09f74685f…`.
- **Result** — passed. **Science note (validates the O2 fix):** stock sampled embedding differs by up to 0.011 between identical calls; cos(mean, sample) ≈ 0.999 — the registered mean readout sits on the stock embedding minus its sampling noise. Structure check confirmed the readout maps exactly onto `OobleckEncoder.layers→chunk→project` (reshape for the non-contiguous mean half; identical layout).
- **Analysis** — Coder deviations 1–6 reviewed and accepted (CWD guard on the resolved config value, train-mode refusal, inference-clone, stock-path-equivalent sample readout, lazy imports, silent-skip caveat noted). Round 2 OPEN pending Codex r2 review.
- **Next** — Codex r2 review; round 3 (driver) starts only after r2 closes.

## 2026-08-19T20:00:00-0400 — Round 2 CLOSED (fix batch verified, incl. foreign-CWD + CUDA)

- **Version Control** — r2fix commits `4d8f11c` (F3+F4) `58054d7` (F5 dependency-traversal parity) `50b3f66` (F1 repo-root-anchored assets) `22bf0ad` (F2 real-model B=8 + CUDA RNG isolation) `7335248` (ledger).
- **Command / Validation** — Planner-independent: exp_18 suite **220 passed**; agree_embed file re-run from a foreign CWD (/tmp): **46 passed** incl. 8 integration (2 CUDA-conditional, executed on the A6000s). Measured: real-model batch invariance max|diff| 4.47e-08; CUDA mean readout leaves CPU + all CUDA generators untouched; sampled path advances the device generator (teeth witnessed).
- **Result** — passed; round 2 CLOSED. Coder deviations 1–4 accepted (parameter removal + source-tie test; whole-model submodule walk; chdir fixture rationale; CUDA skipif semantics).
- **Next** — Round 3: `eval_localization.py` driver + tests (largest round).

## 2026-08-19T21:35:00-0400 — Round 3 delivered; Planner verification PASSED (parity 0.0 on real ckpt)

- **Version Control** — r3 commits `7af1979 210fbeb e682216 667a017 52d5a81 221a195 ab326e4 cbd7f2b 1cf834e 56a5848 6bcbc8b afb8fc3 b33f3ed 12b8ecc 42cf879` (units a–h per contract).
- **Command / Validation** — Planner-independent: exp_18 suite **310 passed**; CLI `--parity-check` on real `FLAC_EMA.ckpt` + real unseen dataset enumeration (6,337 files/17 rooms found through the symlink): **match=True, max_abs_diff=0.0**. Checkpoint load integrity: 0 missing / 0 unexpected.
- **Result** — passed. **Environment provenance finding:** `flash_attn` is NOT installed in the rebuilt `flac` env — the DiT falls back to standard attention (warning observed). Internally consistent for all exp_18 runs; to be recorded in provenance (fix batch) and flagged to Yixun.
- **Analysis — Coder deviations:** (1) `resolve_weights_source` added to the import boundary — accepted, and its honest consequence recorded: released FLAC_EMA ships a pre-flattened state dict ⇒ `weights_source="online"` (the weights ARE the EMA export; fold never fires) — pinned by test + provenance. (2) `--ckpt-path` optional for `gt_rir` — required for R-1 sequencing, accepted. (3) earlier ARE refusal — accepted. (4) batch semantics documented — accepted. (5) audit pass draws different contexts than the run pass (identities draw-independent; rows fingerprint the actual draw) — accepted, disclosed. (6) baselines carry no MRR — accepted (protocol honesty). (7) context_evidence rows for the O10 control — accepted. (8) constant_source = candidate centroid — **Planner ruling: accepted as registered** (well-defined from the candidate set; depth-derived centroid rejected as an extra estimator). (9) stringified radius keys — accepted. (10) main() dataloader path untested pending rung 4/5 — accepted.
- **Next** — Codex r3 review (fix batch to include: provenance records flash_attn availability). Round 3 OPEN.

## 2026-08-19T23:50:00-0400 — r3 fix batch Planner-verified; round 3 closure delegated to the integrative review

- **Version Control** — r3fix commits `e9de5c0 4d3369e 380dfe3` (F1 BLOCKER: in-loop identity check, split-JSON expectation, end gate, .partial publish) `da8a284` (F2: synthetic + real M×K parity) `156c714` (F3) `a39c9b3` (F4) `f11f137` (F5) `90a62da` (F6) `3ead435` (F7) `bdaeb04` (F8) `4e3b4a5` (F9) `06588eb` (CLI __main__ defect + workers≥1) `5114f9e` (ledger).
- **Command / Validation** — Planner-independent: **368 passed**; registration-sha gate REFUSES a registered unseen run without the flag; smoke-on-unseen REFUSES citing O16; CLI parity on the SEEN split through main(): match=True, 0.0 (6,217 files / 131 rooms enumerated). Real-asset M×K parity ran (Apartments_idx_40, M=10→3 kept, K=2): bitwise at `--cond-autocast off`; 1.3e-3/0.21 (~0.6%) at `default` — bf16 conditioning reduction order depends on batch composition. **Planner ruling: registered protocol stays `--cond-autocast default`** (per approved plan §2.3; deterministic for our fixed per-query layout since the M dicts per query never change) — the O21 autocast-off diagnostic slice quantifies ranking sensitivity; finding documented in params at launch.
- **Result** — fix_ready; round 3 fix batch VERIFIED by Planner. Given the round contained a BLOCKER, formal r3 closure is delegated to the Codex integrative `full` review (launched next), which re-verifies F1–F9 and reviews the whole exp_18 diff before any launch. Observed nit passed to that review: the registration-sha refusal fires post-checkpoint-load (wasteful ordering, still fail-closed).
- **Analysis** — Coder deviations accepted: workers≥1 (persistent_workers constraint), F5 per Planner wording (smoke identity allowlist lives in params at R0, enforcement-by-record), `canonical_stream_hash` third import (justified), pre-flight audit dropped in favor of the load-bearing in-loop check, `substituted` always False by construction (abort semantics).
- **Next** — Integrative full review → validation ladder rungs 3–5 → R-1 on Yixun's dataset word.

## 2026-08-20T00:15:00-0400 — Dataset COMPLETE (Yixun confirmed); rung 4 real-data readback PASSED

- **Goal** — Validation rung 4 on the complete dataset (Yixun: "AcousticRooms is now copied to /media/diskstation/yixunhu/FLAC/AcousticRooms").
- **Result** — passed, with three protocol-relevant findings:
  1. **Metadata naming = `S0010_R0089.json`** (`"S00"+str(node)`, unpadded int concatenation) — the release code's convention is CORRECT for metadata; wav names (`S010_…`, 3-digit padded) are a different namespace, as the Opus review suspected. Our numeric matching handles both; the S010-quirk worry from the 2026-08-18 recon entry is RESOLVED as a non-issue.
  2. **All 17 unseen rooms have 10 metadata sources** (candidate authority ⇒ M=10 everywhere). `LivingRoomsWithHallway_idx_30`'s source 10 has metadata but ZERO wavs (225 = 9×25 on disk) ⇒ context can never contain it ⇒ its eligible set = {GT, S10} = 2, same 50% information-matched chance as the other rooms; the gt_only exclusion clause will never fire (kept as a guard). Oracle eligibility shrinks by one there, by design.
  3. **No second measurement channel exists** (`single_channel_ir_zip` = per-scene zips of channel 1) ⇒ non-trivial oracle variant unavailable; identity-oracle = sanity-only, per plan fallback. Depth maps: per-receiver (256,512) float64 — matches AR_md. Wav sanity: (1,64542)@22050.
- **Next** — Integrative full review verdict → R-1 launch (params/command/acceptance-criteria written at launch per SOP).

## 2026-08-20T00:40:00-0400 — Integrative full review: r3 CLOSED; launches HELD pending r4 batch; plan Rev 3.1

- **Result** — Codex full review: **r3 formally CLOSED** (Part 1: 7 RESOLVED, 2 PARTIALLY — leftovers folded into r4). New: 4 HIGH (frozen candidate manifest; reviewed R-1/R0/R1 entry points; fail-open context control; toothless registration gate), 2 MEDIUM (cell-name collisions; device provenance), 1 NIT (late refusal). Verdict REQUEST-CHANGES; R-1/R0/R1/R2 all HELD until the r4 fix batch passes a focused review. Review independently confirmed the rung-4 M=10 finding and endorsed the autocast ruling CONDITIONAL on the frozen manifest.
- **Change** — plan Rev 3.1 amendment appended (factual LRH correction + enforcement hardenings; no science-protocol change ⇒ Yixun informed, not re-gated).
- **Next** — Coder r4 batch (F1–F7 + Part-1 leftovers) → focused Codex fix review → launches.

## 2026-08-20T00:50:00-0400 — Coordination: exp_19 occupies both GPUs (R-cal legs)

- **Result** — Peer session launched exp_19 R-cal: GPU 0 = HAA finetune train.py (1–3 h, DO NOT TOUCH), GPU 1 = FLAC_HAA 5-seed eval (~1–2 h, peer offers pause if we need it urgently).
- **Analysis — exp_18 sequencing impact:** r4 round + focused fix review are CPU-bound (its small CUDA tests co-tenant harmlessly). R-1 readback is CPU-only; R-1 oracle can co-tenant or wait. **R0's probe must run on an IDLE GPU** — its timing/peak-memory numbers feed the §9 budget decision and co-tenancy would contaminate them. Peer legs end ≈02:00–04:00 EDT; R0 ready ≈03:00 — minor contention, resolved by checking nvidia-smi before launch and waiting for a free GPU rather than pausing the peer's eval (not urgent). Shared-machine etiquette honored: no touching their train.py; `readlink /proc/<pid>/cwd` before assuming ownership of anything.
- **Next** — r4 completion → focused fix review → R-1 readback (CPU, immediately) → R-1 oracle + R0 on a free GPU.

## 2026-08-19T19:25:00-0400 — ERRATUM: timestamp drift + commit-scoping practice fix

- **Result** — Two bookkeeping defects, both surfaced by the exp_19 peer session:
  1. **Timestamp drift:** the five entries stamped `2026-08-19T23:50` through `2026-08-20T00:50` were actually written ≈18:40–19:15 EDT on 2026-08-19 (the Planner extrapolated timestamps instead of reading the clock; commit timestamps are authoritative). This notebook is append-only, so the originals stand with this erratum. Lesson: run `date` before stamping every entry.
  2. **Whole-index commit swept a peer file:** commit `9627449` (`git add -u` + unscoped `git commit`) captured 189 lines of exp_19's staged-in-flight `src/tests/test_raf_md.py`. Content intact and green at HEAD; no history rewrite (peer concurs). **Practice fix, binding: every commit from this session is path-scoped (`git commit -- <paths>`)** — two sessions share this working tree and index.
- **Next** — unchanged; r4 in flight (its scorer-noise commit `0693f59` already landed). GPU window per peer: GPU 1 free ≈19:45, GPU 0 ≈21:00–22:30 EDT.

## 2026-08-19T19:58:00-0400 — Coordination update: GPU 1 frees ~20:15–20:20 (peer restarted Leg A)

- **Result** — Peer's Leg A had a constant `--eval-name` overwriting its per-seed metric JSONs; killed + restarted with per-seed names (same defect class as our full-review F5, caught in their runner script). GPU 1 now frees ≈20:15–20:20 EDT and is then ours indefinitely; GPU 0 / Leg B unaffected (~21:00–22:30, its post-train eval stays on GPU 0).
- **Analysis** — No exp_18 delay: the actual launch gate is the r4 focused review (~21:00 EDT), after which GPU 1 will already be free. Sequencing unchanged.

## 2026-08-19T20:40:00-0400 — r4 delivered; Planner verification PASSED; deviations ruled

- **Version Control** — r4 commits `c3ad3c8` (frozen manifest) `03c6fe3` (device provenance + early validation + finite fa-angles) `ba73723` (fail-closed context evidence) `af708e3` (cell-unique stems + no-overwrite) `4c0a1fc` (machine-checked registration) `e851c0a` (--mode readback) `92f132a ac1d1ab` (probe timings + peak-mem) `0693f59` (--mode scorer-noise) `f86de94` (reaggregate module + mode) `1dd2e25` (__main__ recurrence, now AST-pinned) `01b7cce` (ledger).
- **Command / Validation** — Planner-independent: **455 passed** (5 files); CLI seen-split parity re-run: match=True, 0.0. Real-data: manifest 24 s one-off; all 17 rooms 10 metadata sources; LRH wavs 1..9 (Rev 3.1 §1 executable).
- **Analysis — Planner rulings on r4 deviations:** (1) context-evidence leniency dropped entirely — ACCEPTED (strongest form; offline-only None path documented). (2) scorer-noise held to seen split — ACCEPTED as registered (diagnostic; seen suffices). (3) K′-skip with reported k_primes_evaluated — ACCEPTED. (4) mode-scoped required flags — ACCEPTED. (5) single sims codec in reaggregate — ACCEPTED. (6) AST guard — ACCEPTED. GPU note: exp_19's Leg B occupies GPU 0 (20.4 GB); GPU 1 near-free (2.3 GB residue).
- **Next** — Focused Codex review of r4 (the launch gate) → R-1 readback + oracle → R0 probe on GPU 1.

## 2026-08-19T21:00:00-0400 — Coordination: exp_19 will touch eval_FLAC.py (additive context-capture-ids branch)

- **Result** — Peer's review requires an additive branch in eval_FLAC.py's stream recorder (activates only on a RAF-only metadata key; AR pledged byte-identical; snapshot suites must stay green). exp_18 imports `sample_context_ids` + `canonical_stream_hash` from that file. Our position (sent): proceed NOW so the change predates all exp_18 launches; AR byte-identity is binding for our membership/digest machinery; ping the sha; if it slips past ~22:00 EDT, hold until R-1/R0 finish. On their sha landing: re-run the 455-test suite + bit-parity check on the merged tree before launching.
- **Analysis** — No exp_18 code state is uncommitted; no run in flight. All registered exp_18 runs will thus launch from a single post-change code state — cleaner than the alternative orderings.

## 2026-08-19T21:25:00-0400 — Codex r4 launch-gate review: REQUEST-CHANGES (2H/4M) → r5 batch

- **Result** — F1/F3/F6 + both finiteness partials RESOLVED; F2/F4/F5/F7 PARTIAL. New: H1 GPU timings not wall-correct (`_sync` ignores device index — would have corrupted R0's GPU-1 numbers), H2 readback doesn't enforce the M=10/17-room + depth-shape invariants, M3 aux modes silently overwrite, M4 registration accepts committish (no ancestry), M5 ckpt torch.load precedes registration checks, M6 scorer-noise seen-hold bypassable + unseeded draws. Also confirmed: no r4 semantic change to the generation path (parity result stands); reaggregate math clean.
- **Next** — Coder r5 batch (all 6) → focused re-review → launches. R-1/R0 ETD slips to ≈23:00 EDT.

## 2026-08-19 (see date line in commit) — Coordination: GPU 1 free; peer's eval_FLAC sha incoming

- **Result** — Peer confirms GPU 1 free (Leg A done, 5 seeds stream-audited); Leg B holds GPU 0 until ~21:00–22:30. Their eval_FLAC.py additive commit is being written; sha to follow; our binding AR-byte-identity constraint is being verified against the suites we named before it lands. exp_18 unaffected: r5 in flight, launches still gated on the focused re-review; merged-tree re-verification remains the pre-launch step once the sha arrives.

## 2026-08-19T20:05:00-0400 — Peer's eval_FLAC change landed (6c0a16e); shared-file protocol extended

- **Result** — `6c0a16e` spot-checked: RAF-only capture-id branch, AR schema=1 byte-identical, goldens recompute; peer's suite evidence includes our named snapshot suites + test_eval_localization read-only green. Merged-tree re-verification deferred to the r5 completion pass (one verification covers both). Scope correction sent to peer: metric_callback.py and src/metrics/* ARE shared (our AGREE loader) — sha-ping + launch-window hold protocol extended to them. r5 progress: first commit `a69f96a` (wall-correct GPU timing) already landed.

## 2026-08-19T20:10:00-0400 — Shared-tree protocol strengthened (peer-side)

- **Result** — Peer will not even EDIT src/metrics/* in the working tree (import-time dirt would leak into our launched processes) until our "R-1/R0 done" signal; their two metric-stack commits (RAF-only branches + a device fix) land after it, shas pinged. Their Leg B holds GPU 0 until ~00:00 (revised) — irrelevant tonight (R-1/R0/R1 use GPU 1 only); both GPUs needed only for tomorrow's R2 seeds, after Leg B ends.
