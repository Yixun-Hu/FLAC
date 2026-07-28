# Lab notebook — exp_07_fa_scratch

## 2026-07-07T14:35:55-04:00 — scaffold
- **Goal** — Route B: from-scratch fa_invariant training; commissioned by Yixun (hypothesis on optimizer-state/EMA-only release recorded in the query file with the Planner's evidence note).
- **Version Control** — branch check-equivariance-necessity, base_commit 9ef9003 (end of exp_06, pushed).
- **Resource status at scaffold** — GPU 1 fully free (48 GB); GPU 0 carries Yixun's own relaunched oneroom job (~20.7 GB, PID 2667300) — untouched.
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval (incl. the budget/hardware decision) → runs.

## 2026-07-10T23:50:00-04:00 — pre-launch config-identity audit (Yixun's go-condition) + reviewer-model change

- **Author** — Fable 5 (main session; planning seat restored from Opus 4.8 after Yixun's `/model` switch — exp_08 closure artifacts remain Opus-attributed).
- **Trigger** — Yixun: exp_07 worth going, conditional on confirming B-F ≡ B-V ≡ FLAC-as-described (data/training/model). Full audit: `fa_scratch_config_identity_audit.md`; probes `probe_released_ckpt.py` (+ `..._23:35:28_ckpt_probe.log`), `assert_arm_configs.py` (+ `..._23:43:26_arm_asserts.log`).
- **Headline finding** — the released `FLAC.ckpt` records its own recipe: **effective batch 64, accum 1** (loop-counter proof: accum ratio 1.0000; 4,550 steps/epoch = floor(291,210/64); epoch 14 @ step 67,500) — *[corrected per review H1: the counters prove the GLOBAL eff batch + accum only; the **micro-64 × 1-GPU decomposition is paper-specified** (§B.1), 2×32×1 being counter-indistinguishable]* — AdamW 5e-5/(0.9,0.999)/wd 1e-3, InverseLR(1e6, 0.5, 0.99) with `_last_lr` matching the analytic value, EMA on, ModelCheckpoint every 2,500 (release = periodic checkpoint, mid-epoch 15). **Corrects the plan's eff-128 assumption; wall-clock halves → B-V ~5 d, B-F ~16.7 d.** Plan revised (dated §REVISION).
- **Identity verdicts** — B-F ≡ B-V PROVEN (byte-copy + 2 training keys; instantiation asserts: identical 64.50M/753-tensor architecture); B-V ≡ released config PROVEN up to no-op keys (ckpt-embedded `model_config` diff itemized — all metrics/demo/dead-option/default keys; `structured_noise:false` and never consumed); B-V ≡ paper text verified on every stated number; ⚠ paper-text unseen count 5,244 vs shipped 6,337 (shipped split authoritative per exp_01 + announcement 01). Yixun's GPU-0 `FLAC_vanilla291k`: NOT certifiable as B-V (folder `single_channel_ir` ≠ `single_channel_ir_1`, micro 16×4, third-party file copies) — exp_07 runs its own B-V; optional corroborating screen of its step-67,500 ckpt noted.
- **Reviewer-model change (Yixun)** — Codex reviewer is now **`gpt-5.6-sol` at xhigh** ("extra high"), superseding `gpt-5.5`. Required a CLI upgrade: 0.142.5 → **0.144.1** (native tarball → `~/.local/share/codex-native-0.144.1/`, symlink swapped; old install retained for rollback); model probe returned OK. SOPs updated: `worklog/experiment_SOP.md` (this repo), `rir2rir/worklog/experiment_SOP.md`, `rir2rir/worklog/code_migrate_SOP.md` (the rir2rir edits are uncommitted in that repo).
- **Next** — consolidated Codex (`gpt-5.6-sol` xhigh) review of the two probe scripts + arm configs + audit numbers → fixes → commit; then Yixun's revised-budget approval → `--max-steps` TDD round → M0 probes → launch when a GPU frees.

## 2026-07-11T00:05:00-04:00 — audit review round: REQUEST-CHANGES → all findings fixed (one rebutted with evidence)

- **Review** (`fa_scratch_codex_code_audit_probes_review.md`, first `gpt-5.6-sol` xhigh use): 2 Blocking, 3 High, 3 Medium, 4 Low — the sharpest review to date; every finding verified before acting.
- **Fixes applied:**
  - *Blocking 1 (inline unreviewable diff, no command file):* config-diff moved INTO `probe_released_ckpt.py` (v2, rerun → `..._23:59:06_ckpt_probe_v2.log` supersedes v1); `fa_scratch_command.md` created with a post-hoc provenance-deviation record.
  - *Blocking 2 (asymmetric micro×accum fallback):* rule rewritten — both arms run the identical pair chosen from **B-F's** fit constraint (64×1 → 32×2 → 16×4); asymmetric = declared ablation only.
  - *High 1 (counter phases / decomposition):* v2 probe prints all phases; accum computed as **micro(`processed`) / optim(`completed`)** = 67,500/67,500 = 1.000000 — like-for-like at save time, since the optimizer-step `completed` counter increments before the checkpoint callback fires while the micro-batch `completed` counter lags by one (only the micro `completed` phase is lag-affected); steps/epoch 4550.0 exact; audit softened — eff-64+accum-1 counter-proven (conditional on shipped split + drop_last), **64×1-GPU decomposition paper-specified** (2×32×1 counter-indistinguishable). README example (eff 128) adjudicated: ckpt-incompatible on two counters — provenance of the old plan assumption.
  - *High 2 (ViT initializer):* reclassified training-relevant; verdict downgraded to "config-identical up to no-op keys + one initializer-provenance caveat"; our pin recorded (rev `114c1379…`, sha256 `4610ad75…`), identical across arms; unknowable v added.
  - *High 3 (unadjudicated launch settings):* explicit launch manifest added to audit §6 — every non-recoverable setting labeled a CHOICE (workers 6, val-every −1 with RNG-purity rationale, seed 42, matmul policy recorded at launch), identical across arms.
  - *Medium 1 (asserts too weak):* `assert_arm_configs.py` v2 — wrapper-field asserts, real `configure_optimizers()` object checks (caught the InverseLR step-0 warmup lr `5e-7` — assert corrected to closed form; red→green logs kept), **seeded init-identity: state_dict sha256 match across arms** (`44a2f6aa…`). Rerun green (`..._00:00:17_arm_asserts_v2.log`).
  - *Medium 2 (plan contradictions):* plan lines on "≥2 GPUs" and "10k cadence" explicitly corrected with strikethrough + audit cross-reference.
  - *Medium 3 (BN claim):* **REBUTTED with evidence** — 20 `BatchNorm2d` modules exist in the instantiated model (`context_audio.net.cnn.*`, torchvision resnet18; invisible to source grep; the very modules exp_05/08's FreezeBN froze). BN caveat retained; reviewer's additional caveats (accum summation order, bf16 reduction, RNG sequencing, data order) adopted alongside.
  - *Low 1–4:* full InverseLR formula shown (warmup term ≡ 1.0 at 67.5k; exact-digit match); `img_h/w` explanation corrected (SimpleViT no-HF-path branch); dataset row split into total (302,671/260) vs train (291,210/243); this worklog's prior timestamp given its UTC offset.
- **Next** — focused re-verify review of the fixes → commit + push → Yixun's revised-budget approval.

## 2026-07-11T00:15:00-04:00 — re-verify round 2: 7 RESOLVED + BN rebuttal ACCEPTED; 5 residuals + 1 new finding → all fixed

- **Re-verify** (`fa_scratch_codex_code_audit_probes_reverify.md`): M1/B2/L1–L4 RESOLVED; **M3 rebuttal ACCEPTED** (reviewer independently confirmed the 20 BatchNorm2d modules via `conditioners.py:37`). Residuals fixed:
  - *B1:* audit evidence-sources line now cites the **v2 probe log as canonical** and marks v1 superseded with the provenance-deviation pointer.
  - *H1:* worklog headline entry corrected in place — counters prove global eff-64+accum-1; micro-64×1-GPU is paper-specified.
  - *H2 (reviewer's single-most-important):* pin made **fail-closed** — `assert_arm_configs.py::assert_vit_pin()` hard-fails unless the HF cache holds exactly rev `114c1379…` with sha256 `4610ad75…`; launch manifest adds `HF_HUB_OFFLINE=1`. v3 asserts run green (`fa_scratch_2026-07-11_00:11:31_arm_asserts_v3.log`).
  - *H3:* manifest completed — `--val-dataset-config` omitted entirely (val loader conditional, `train.py:52-58`), gradient-clip/strategy relabeled **choice**, deps row expanded (torch/PL/transformers pins + full `pip freeze` at launch).
  - *M2:* plan swept — §1 eff-128 line, §2 budget table (now 5.0/16.7/21.7 d), options (b)/(c) wall-clocks, and the recommendation line all corrected with strikethroughs.
  - *NEW-M:* worklog accum wording corrected to **micro(`processed`)/optim(`completed`)** with the save-time like-for-like explanation.
- **Round closure** — a terse third Codex pass (`..._reverify2.md`) verifies the six fix diffs (incl. the new `assert_vit_pin` code); commit follows its verdict.

## 2026-07-11T00:18:00-04:00 — reverify2: 5/6 RESOLVED; last item fixed with red/green proof → ROUND CLOSED

- **reverify2 verdict:** B1/H1/H3/M2/NEW-M RESOLVED; H2 NOT-RESOLVED with two new defects in `assert_vit_pin`: hard-coded cache path ignoring `HF_HOME`/`HF_HUB_CACHE`, and `assert` statements stripped under `python -O`.
- **Fix (v4):** cache root resolved via `huggingface_hub.constants.HF_HUB_CACHE` (the same resolution the transformers loader uses) and all gate checks converted to explicit `raise RuntimeError`. **Proof in `fa_scratch_2026-07-11_00:16:07_arm_asserts_v4.log`:** green run exit 0 (pin OK, all asserts pass); RED test 1 — `HF_HUB_CACHE` pointed at an empty dir → `RuntimeError`, exit 1 (env honored, fail-closed); RED test 2 — same under `python -O` → still raises, exit 1.
- **Closure basis (SOP: "re-review or Planner verification"):** this final fix is mechanical, its failure modes are the two the reviewer named, and both are red/green-proven in the committed log — closed on **Planner verification**; the files re-enter Codex review at exp_07's next round (`--max-steps` TDD) regardless. Known residual (accepted): the *arm-comparison* asserts in the script body still use `assert` (stripped under `-O`) — only the security pin gate needed `-O` survival; the script is never invoked with `-O` in our flow.
- **Result — audit round CLOSED.** Verdicts stand: B-F ≡ B-V proven (init-identical, 2-key diff); B-V ≡ released config up to no-op keys + pinned-initializer caveat; eff-batch-64 correction adopted throughout; launch manifest final. Awaiting Yixun: revised budget approval (hybrid (c): B-V ~5 d → gate → B-F ~16.7 d), GPU window, optional 291k corroborating screen.

## 2026-07-11T00:35:00-04:00 — GO from Yixun; TDD round 1 (--max-steps) started

- **GO** — all five staged steps approved verbatim (query file Q4): TDD → M0 → B-V 67.5k → 2σ gate (stop-and-ask if fail) → B-F. GPU 1 held for the ~3-week sequential window (currently free — the sb_cvae job ended). Corroborating screen approved context-only.
- **Corroborating screen DEFERRED** — Yixun's `FLAC_vanilla291k` is at step ~45,000 (67,500 ckpt does not exist yet; ~1.8 d away at its pace). Re-scheduled into the step-4 gate block (~2026-07-16), which is where its context matters and where GPU 1 has its natural B-V→B-F gap.
- **Round 1 open** — Opus 4.8 max Coder spawned with a tight brief: `--max-steps` flag (defaults.ini `max_steps = 1000000` + `train.py:161` de-hardcoding), test-first in `src/tests/`, minimal extraction per the `finetune_cond.py::build_trainer_kwargs` precedent, red→green + full-suite non-regression, <200 lines, no commit (main session commits post-review). Reviewer on deck: Codex `gpt-5.6-sol` xhigh.
- **Next** — Coder returns → Codex review → fixes → commit → M0 probes (micro ladder 64→32→16, BOTH arms, EMA on) → B-V launch.

## 2026-07-11T01:10:00-04:00 — round 1 code landed; review REQUEST-CHANGES (1M+1L); fixes dispatched

- **Coder round 1 result** — red (8 failed, intended reasons) → green (8/8 new + full suite 119 passed); Planner independently re-ran: 119 passed. Design: full `build_trainer_kwargs` extraction with a byte-for-byte kwargs pin against the old inline literal (Option A per brief). Diff: train.py +32/−14, defaults.ini +3, new `src/tests/test_train_max_steps.py` (168 lines).
- **Review** (`fa_scratch_codex_code_max_steps_review.md`, gpt-5.6-sol xhigh) — REQUEST-CHANGES: *Medium* — no Trainer-boundary test (all 8 tests pass even if `main()` reverts to the inline literal); fix = `construct_trainer` wrapper + monkeypatched `pl.Trainer` capture + revert-guard. *Low* — CLAUDE.md HAA section still forbids the flag (stale; CLAUDE.md is local/gitignored). Confirmed clean: kwargs faithfulness, prefigure int parsing, no `finetune_cond.py` collision (independent argparse, default 2000), conftest stale-src guard.
- **Fixes dispatched to the same Opus Coder** (context preserved) with test-first instructions.
- **Commit/push HOLD** — Yixun requested a commit/push inventory + approval gate (proposal delivered: one FLAC round commit post-fix; 2-file surgical rir2rir SOP commit; leave the other session's 43 rir2rir items alone). Awaiting his word; M0 also holds (round-closure: no launch while the round is open).

## 2026-07-11T01:30:00-04:00 — round-1 fixes verified; round functionally CLOSED (commit gated on Yixun's approval)

- **Coder fix round** — `construct_trainer` wrapper (train.py:50; main() calls it at :182); boundary test (monkeypatched `pl.Trainer` capture, asserts `max_steps=67500` reaches the constructor exactly once); revert-guard test (regex catches kwarg-form, dict-key-form, and re-hardcode reverts; verified to spare the legit wiring); CLAUDE.md HAA section rewritten to prescribe `--max-steps 1000` (local file, gitignored). Red 3-failed (intended) → green 10/10; full suite **121 passed**.
- **Planner verification (closure basis)** — independent full-suite re-run (121 passed); call-site wiring inspected; adversarial regex re-derivation: all three revert vectors bitten, legit line spared. Both review findings implement the reviewer's own prescribed shapes → closed on Planner verification; the files re-enter Codex review at the next round regardless.
- **Round 1 CLOSED** pending the approved commit. Staged next: FLAC round commit (+push) → rir2rir 2-file SOP commit (+push) → M0 → B-V.

## 2026-07-11T14:20:00-04:00 — commits/pushes approved & executed; M0 ladder exhausted → documented amendment (extend to 8×8/4×16)

- **Commits (Yixun-approved):** FLAC round-1 `e85ebde` + bookkeeping, pushed (remote sync). rir2rir SOP 2-file surgical commit `4425497`, pushed; the other session's in-flight work untouched (it committed `6e1a270` on top seconds later — left local, its call).
- **M0 (as registered, 3 rungs) — NO FIT for B-F:** 64×1, 32×2, 16×4 all `torch.OutOfMemoryError` inside `invariant_conditioning` → DINOv3 forward (probe process alone held ~47.3 GiB each time; peaks 48.1/48.2 GiB). Red-herring note: tracebacks say "GPU 0" due to CUDA renumbering under `CUDA_VISIBLE_DEVICES=1` — the probe process's ~47 GiB sole ownership proves physical GPU 1; the 744 MiB "peak" on the 16×4 attempt is a 5-s sampler-cadence artifact (sampler tightened to 1 s for the extension).
- **AMENDMENT (documented, not silent):** the audit §6 rule's 3-rung list was exhaustive as written; it is hereby extended down two eff-64-preserving rungs — **8×8 → 4×16** — same-pair-both-arms and eff-batch-64 (the constraints Yixun's GO actually pinned) unchanged. Precedent that it should fit: exp_08's A-F trained at micro 4 (×32) on this GPU, EMA off; EMA adds only a ~0.26 GB model copy. If 4×16 also OOMs → hard stop, ask Yixun.
- **Next** — extended ladder → B-V confirm at winner → throughput → B-V launch.

## 2026-07-11T14:55:00-04:00 — M0-ext VERDICT: common pair 8×8; B-V LAUNCHED

- **M0-ext:** B-F @ 8×8 **fits** — exit 0, peak 36.8/49.1 GiB, 0.65 micro-it/s → **5.2 samples/s** (anchor was 3). B-V @ 8×8 confirms — peak 10.5 GiB, 1.86 micro-it/s → **14.9 samples/s** (anchor was 10). Consistency: 36,401 micro/epoch = floor(291,210/8) ✓; live `lr=7e-6` @ opt-step 15 = (1−0.99¹⁵)·5e-5 ✓ (InverseLR warmup, matching the asserts); `--max-steps 15` stopped exactly ✓ (flag's first production use).
- **Common pair pinned: micro 8 × accum 8 (eff 64) for BOTH arms** per the amended §6 rule. Re-anchored wall-clock: **B-V ≈ 3.4 d, B-F ≈ 9.6 d, total ≈ 13.0 d** (was 21.7 d).
- **M0 docs committed `70dea5a`, pushed** (SOP commit-before-long-run).
- **B-V LAUNCHED** (command in `_command.md` at launch; log `fa_scratch_*_BV_train.log` with env records: torch matmul/TF32 flags + pip freeze header; pre-launch fail-closed pin gate in-script). Target 67,500 opt steps, ckpt every 2,500. Screens: external eval_FLAC at 10k marks (first ≈ 09:00 Jul 12); `use_ema=false` eval-config copy to be committed before the first screen.
- **Projected gate: ~Jul 15** (B-V ends ≈ late Jul 14 + gate evals) — a day earlier than the pre-M0 estimate.

## 2026-07-12T—worklog namespace migration (Yixun directive; announcement 03)

- **Change** — repo-wide `git mv`: everything under `worklog/` → `worklog/worklog_yixun/` EXCEPT `worklog/experiment_SOP.md` (stays, per Yixun). SOP directory-layout section + all path examples rewritten to the `worklog_<username>/` convention; CLAUDE.md + memory updated; announcement `03_worklog_username_namespace.md` records the standing rule.
- **Safety for the LIVE B-V run** — the tee'd training log follows its inode through a same-filesystem rename: verified growing at the new path post-move. Configs were read at startup; checkpoints go to `outputs_FLAC/exp07_BV` (outside worklog). B-V untouched (GPU 1 @ 100%).
- **Script repairs (5, mechanical)** — fixed-depth `REPO = dirname³` computations replaced with a `.git` marker-walk (layout-proof) in: exp_08 `aggregate_results.py`, exp_07 `assert_arm_configs.py` (+ self-locating `HERE`), exp_05 `dispersion_check.py`, exp_06 `s3_probes.py`, exp_02 `gen_visuals.py`. Verified: aggregator runs; **fail-closed gate re-ran green at the new depth with the SAME init-identity hash `44a2f6aa…`** (move perturbed nothing); remaining three py_compile OK. These mechanical edits are batched into the next round's Codex review per the universal-review batching rule.
- **Note** — historical `_command.md` files keep their original (pre-move) paths as faithful records of what was run; all FUTURE commands use `worklog/worklog_yixun/...` paths (e.g. the B-F launch will reference the moved `FLAC_AR_BF.json`).

## 2026-07-12T09:15:00-04:00 — screen S=10,000 (15% of budget): trajectory healthy; late-field already near released K=1 level

- **Result (K=8, full unseen split, bf16; EMA / online):** T60 **9.82 / 10.20**, C50 1.298 / 1.301, EDT 48.68 / 48.28, FD 0.373 / 0.368, R@1 1.91 / 1.99. (JSONs in `outputs_FLAC/exp07_BV/`.) Note: eval batch 64 (defaults.ini) → 100 eval batches vs exp_01/08's 32/199 — metrics are per-sample accumulations, protocol-equivalent; gate evals will match exp_01 exactly.
- **Reading vs released Table-1 K=8 (8.609/0.968/37.10/R@1 7.06):** at 15% of training, T60 is within 1.2 pp of target and already better than the released *K=1* level (9.97); EDT (+11.6 ms), C50 (+0.33) and especially retrieval R@1 (1.9 vs 7.1) lag — the expected shape early in from-scratch training (late-field statistics converge first; early-field detail and retrieval-grade specificity come late). EMA vs online: EMA ahead on T60 by 0.38 (smoothing already helping). **No pathology; no action.**
- **Co-location cost negligible** — B-V dipped 1.94 → 1.90 it/s during the eval, loss 0.471 ↓ after. Next screen at S=20,000 (~ETA +4.5 h), same protocol.

## 2026-07-12T13:30:00-04:00 — screen S=20,000 (30%): T60 within 0.31 of released target; all metrics improving steeply

- **EMA @ 20k (K=8 full split):** T60 **8.92** (10k: 9.82 → Δ−0.90), C50 1.094 (−0.20), EDT 45.62 (−3.07), FD 0.333 (−0.040), R@1 3.74 (+1.83). Online screen running. Trajectory table: T60 gap-to-released-target 1.21 → **0.31** pp between 15% and 30% of budget — on a clean convergence path toward the gate band; EDT still the laggard (45.6 vs 37.1), retrieval halfway (3.7 vs 7.06).
- **Watcher nit (accepted):** the persistent screen watcher's one-line summary extraction is empty (its grep window lands in tqdm output) — evals + logs + notifications all work; I extract numbers from the tee'd log on each ping. Not worth a mid-loop restart; noted for the next-round review batch.
- **No action needed** — nothing pathological; next screen S=30,000 ≈ +4.5 h.
- **Online @ 20k (appended):** T60 9.16, C50 1.125, EDT 47.77, FD 0.330, R@1 3.42 — EMA leads online on every perceptual metric (T60 by 0.24, EDT by 2.2); the EMA-vs-online gap is *narrowing* as training proceeds (T60: 0.38 @ 10k → 0.24 @ 20k), consistent with EMA being a smoothed trailing average rather than a divergent branch.

## 2026-07-13T01:20:00-04:00 — screen S=30,000 (44%): **T60 CROSSES BELOW the released target** (8.34 < 8.609)

- **EMA @ 30k:** T60 **8.3375** (< released 8.609 — the first model in this whole program, fine-tunes included, to beat the released T60), C50 1.0116 (gap 0.044), EDT 40.96 (gap 3.86, still closing steadily), FD 0.3174 (≈ at target), R@1 4.04 (target 7.06).
- **Interpretation** — the lineage question (exp_03–06's blocker: is the released operating point reachable on our data/env?) is resolving to **yes** in real time: T60 and FD are at/below target at 44% of budget; EDT/C50/R@1 still converging with 56% of budget left. Also notable: our trajectory *at 30k* already beats the released 67.5k checkpoint's T60 — compatible with exp_06's surviving "checkpoint selection" hypothesis (the release may not be the trajectory's best point).
- **Yixun notified by push** (remote control armed); no action needed. B-V at epoch 6, loss 0.539; next screen S=40,000.
- **Online @ 30k (appended):** T60 **8.1399** (below EMA's 8.34 — the lead flipped: on a steep still-improving trajectory the live weights lead and the trailing EMA lags; EMA re-takes the lead when the curve flattens), C50 1.0092, EDT 40.89, FD 0.3184, R@1 3.85. Both weight streams now beat the released T60 at 44% of budget.

## 2026-07-13T12:55:00-04:00 — screen S=40,000 (59%): T60 back AT target (8.60); C50/EDT slopes flattening ABOVE target → strict-2σ gate projection tightens

- **EMA @ 40k:** T60 8.6027 (bounced 8.34 → 8.60; ≈ released 8.609 exactly; the 0.26 swing is ~20× eval-σ → real training fluctuation around the target, not eval noise), C50 1.0047 ↓, EDT 39.99 ↓, FD 0.3111 ↓ (at target), R@1 4.83 ↑. Loss 0.390 ↓.
- **Honest gate projection (pre-registered 2σ bands are eval-seed-only, hence brutally tight: T60 ±0.024, C50 ±0.006, EDT ±0.14, R@1 ±0.20):** T60/FD are AT target and plausibly inside/near band at 67.5k. But per-10k slopes are flattening above target for **C50** (−0.204, −0.082, −0.007 → sitting ~1.005 vs needs ≤0.974) and **EDT** (−3.06, −4.66, −0.97 → ~40.0 vs needs ≤37.24); **R@1** 4.83 vs needs ≥6.86. Barring a late plunge, the strict gate likely reads **T60/FD PASS + C50/EDT/R@1 FAIL** → per Yixun's step-5 instruction, that outcome = **STOP and ask**, with the residual-gap quantification + trajectory slopes + a checkpoint-selection curve (we hold 2.5k-cadence ckpts) prepared for the decision. Same statistical caveat as exp_08: single-training-seed bands; the released point is one draw of its own seed/env.
- **No action now** — the projection is logged so the gate report can't be accused of hindsight; remaining screens (50k, 60k) will sharpen the slopes.
- **Online @ 40k (appended):** T60 8.8957 (EMA 8.6027 — EMA re-took the T60 lead as the curve flattened, as predicted at 30k), C50 1.0055, EDT 40.96, FD 0.3094, R@1 4.78. EMA is now the better stream on T60/EDT — supports gating on EMA weights, which is also the released artifact's own convention.

## 2026-07-14T00:35:00-04:00 — screen S=50,000 (74%): C50 crosses 1.0 toward band; T60 oscillation band brackets the released point

- **EMA @ 50k:** T60 8.9826 (trajectory 9.82→8.92→8.34→8.60→8.98 — an oscillation band ≈[8.3, 9.0] that **brackets the released 8.609**), C50 **0.9858** (first time <1.0; gap to band edge 0.974 now just 0.012), EDT 40.34 (≈flat around 40 since 30k), FD 0.3221 (mild bounce from 0.311), R@1 5.25 (steady climb; needs 6.86).
- **Mechanistic note** — InverseLR barely decays this run: lr@50k ≈ 4.88e-5 (97.6% of base; the original's lr@67.5k was 4.84e-5). Training stays "hot" through the end by design → metric oscillation through 67.5k is expected, and the released 67.5k point plausibly sits INSIDE such a band — sharpening the checkpoint-selection interpretation the gate package will quantify (2.5k-cadence curve).
- **Timing update** — B-V completion projects ≈ **09:00–10:00 Jul 14** (screens S=60k ≈ 05:00), gate evals (5 eval-seeds × K∈{1,8}, EMA + the 291k corroborating screen — its step-67,500 ckpt should exist by now) ≈ 3–4 h after → **gate package lands ~afternoon Jul 14**, earlier than the Jul-15 estimate.
- **Online @ 50k (appended):** T60 9.3382, C50 0.9975, EDT 41.76, FD 0.3287, R@1 4.94 — online swings wider than EMA on every metric under the still-hot lr; EMA confirmed as the gate stream (also the release's own convention).

## 2026-07-14T14:00:00-04:00 — screen S=60,000 (89%): **C50 lands IN the 2σ band**; EDT's stall reversed; gate now genuinely open

- **EMA @ 60k:** T60 9.1513 (oscillation band now [8.34, 9.15], still bracketing the released 8.609 — the 67.5k endpoint is a draw from this band), **C50 0.9679 — AT the released 0.9682, inside the ≤0.974 band ✓**, **EDT 38.29** (the 40k–50k stall broke: −2.05 in this interval; needs ≤37.24 — extrapolates to ~36.8–37.6 at 67.5k, i.e. borderline-in), FD 0.3142, R@1 5.76 (advisory; needs 6.86, climbing ≈+0.5–0.9/10k → lands ~6.2–6.5, close-but-short).
- **Revised gate outlook (third update, logged pre-outcome):** C50 in-band ✓; EDT borderline; T60 = endpoint draw from a band that brackets target (the 2.5k-cadence selection-curve will matter); R@1 advisory per plan §3. The strict verdict is now genuinely open — anywhere from PASS to mixed-borderline. All three outlook updates (optimistic @30k → pessimistic @40k → open @60k) are logged as written, so the gate report carries its full prediction history.
- **Timeline correction** — actual pace since 50k was slower than the morning projection: B-V is at ~60.1k/67.5k at 14:00 → completion ≈ **22:45–23:30 tonight**; gate evals (~3–4 h: 5 eval-seeds × K∈{1,8} EMA + 291k corroborating screen + selection-curve extras) → **gate package ≈ 02:00–04:00 Jul 15**. My earlier "morning Jul 14" completion estimate was wrong (based on an unstable early-epoch it/s reading); corrected here.
- **Online @ 60k (appended):** T60 9.2067, C50 0.9677 (also in-band), EDT 38.58, FD 0.3109, R@1 5.67. Screen watcher completed its full S=20k–60k sequence cleanly.
- **291k corroborating ckpt located** — `epoch=14-step=67500.ckpt` exists in Yixun's run dir. The other session's own eval of it (T60 3.35, C50 0.667, EDT 34.58; no FD/R@k, no meta sidecar) is **protocol-incomparable** (different split — plausibly seen/val — different code copy, unknown K); footnote only. The gate-block corroborating screen runs THEIR ckpt under OUR full protocol (unseen 6337, our eval_FLAC, bf16, K=8 seed 42) as approved.
- **Gate block staged** (launches on B-V completion notification): (1) final-ckpt gate evals K∈{1,8} × eval-seeds 42–46, EMA; (2) 291k corroborating screen; (3) selection-curve extras — K=8 seed-42 EMA screens at 2.5k-cadence points bracketing the T60 band (62.5k, 65k, 67.5k + the 27.5k/32.5k neighborhood of the 30k minimum); (4) verdict computation vs exp_01 released 5-seed stats (tiered σ_c, mirroring exp_08); (5) package + push.

## 2026-07-14T21:45:00-04:00 — B-V COMPLETE at exact recipe parity; gate block launched; verdict script under pre-data review

- **B-V endpoint** — `Trainer.fit stopped: max_steps=67500 reached`, exit 0. **Endpoint parity with the released ckpt's own records: epoch 14 ✓, lr 4.84e-5 ✓** (release recorded epoch=14, `_last_lr` 4.8393e-5). 27 checkpoints at 2.5k cadence. Final train loss ~0.434. Wall ≈ 3 d 7 h (vs 3.4 d anchor).
- **Gate block launched** (command in `_command.md`): 15 sequential evals on GPU 1 — gate K=8 × seeds 42–46, gate K=1 × seeds 42–46 (EMA, final ckpt), 291k corroborating screen (THEIR ckpt, OUR protocol), selection-curve extras S∈{27.5k, 32.5k, 62.5k, 65k}. ETA ≈ 2.5–3.5 h.
- **`gate_verdict.py` written and sent for PRE-DATA logic review** (gpt-5.6-sol xhigh) while the evals run — verdict math (ddof-1, σ_c, tiers, 6/6 rule), filename-pattern correctness vs eval_FLAC output convention, and the hardcoded exp_01 baseline stats all get checked before the script ever touches data. Universal-review rule honored ahead of the decision moment.
- **Next** — evals land → apply review fixes → run verdict → assemble gate package → push Yixun (step-5 stop-and-ask if not 6/6).

## 2026-07-14T22:20:00-04:00 — gate_verdict.py review: CONDITIONAL PASS (no gate-blocking error); all 4 findings fixed pre-data

- **Review** (`fa_scratch_codex_code_gate_verdict_review.md`): core verified clean — the reviewer independently recomputed exp_01's raw per-seed stats and confirmed every hardcoded baseline; tier math, directions, 6/6 rule, K-pattern disambiguation all correct. Findings fixed: *Medium* 291k lookup → exact expected path, no wildcard discovery; *Low* fail-closed validation → explicit RuntimeError + isfinite on all primary metrics (assert removed); *Low* selection curve → step-keyed dict, conflicting-duplicate rejection, expected-points warning, regex-anchored parsing; *Low* TIE branch added.
- **Verification** — compiles; red dry-run fires the fail-closed path exactly as designed (refuses on the not-yet-written seed-43 JSON with a precise message). Gate block at 1/15 evals (~00:15 ETA).

## 2026-07-14T23:40:00-04:00 — GATE VERDICT: strict FAIL 1/6 → STOP-AND-ASK (per GO step 5). Decomposition: endpoint-draw + systematic EDT gap, corroborated as lineage by the 291k run

- **Strict gate (B-V@67,500 EMA, 5 eval-seeds, tiered σ_c):** 1/6 primary cells in-band. K=8: T60 9.509 (+0.90, 65σ_c), C50 0.9746 (**misses by 0.0001 dB** — d/σ_c=2.04 vs cutoff 2.0), EDT 42.75 (+5.65). K=1: T60 10.513 (+0.54), **C50 1.0330 PASS (SUPERIOR, −1.5σ_c)**, EDT 44.65 (+4.70). Advisory R@1: 6.18/6.04 vs 7.06/6.83 (closing all run).
- **Decomposition (the honest story):**
  1. **T60 = endpoint draw.** Selection curve (11 points): band [8.34, 9.52] since 20k with the released 8.609 **INSIDE it**; the 67,500 endpoint (9.52) is the band MAXIMUM — worst-possible draw. Steps 30k–40k sat at 8.34–8.60 ≈ target.
  2. **EDT = real systematic gap.** Even the band's best points carry +2.5–3 ms vs released (37.10); endpoint +5.6. Not checkpoint luck.
  3. **C50 = at target** (band 0.968–1.01; K=1 superior; K=8 out by 1e-4 dB).
  4. **R@1 = climbing to the end** (1.9→6.2), best at 65k–67.5k; likely budget-limited and/or lineage.
- **291k corroboration (THEIR independent from-scratch run @67.5k, OUR protocol):** T60 9.92, **C50 0.9673 ≈ released exactly**, EDT 40.75 (+3.7), R@1 6.83. **Same signature as B-V** (C50 at target; EDT/T60/R@1 short) from a different data variant, micro-batch, and environment ⇒ the residual gap is **systematic lineage (data/simulator version), not our run's bug** — exactly exp_06's surviving explanation, now with two independent corroborating runs. (Supporting evidence: paper text says 5,244 unseen items; the shipped split has 6,337 — dataset-version drift is documented.)
- **STOPPED as instructed.** GPU 1 idle; B-F NOT launched. Decision package → Yixun with options (proceed-reframed / extend / investigate / stop) and a symmetric checkpoint-selection pre-registration for any B-F comparison.

## 2026-07-15T—B-V PARITY MANDATE (Yixun Q5): folder-variant hypothesis KILLED; parity program P0 launched

- **Yixun's decision** — before B-F: B-V must reach the released numbers; deliver causal analysis + close the gap.
- **Forensics (audit §7 CORRECTION):** `single_channel_ir` (291k root) ≡ `single_channel_ir_1` (ours) — 302,671 files each, sampled md5s identical. The folder-name "data difference" I flagged in the audit was cosmetic. **Consequence:** our B-V and the 291k run trained on IDENTICAL data → their profile differences (EDT 42.75 vs 40.75, R@1 6.18 vs 6.83 at the same step, same eval) are attributable to **micro-batch (8×8 vs 16×4), seed/env, endpoint draw** — and both runs' shared EDT/R@1 shortfall vs the release points at factors shared by both-but-not-the-release: **micro < 64, hub DINOv3 init, and/or the authors' internal data version** (5,244-vs-6,337 paper-text evidence).
- **Suggestive direction:** larger micro correlates with better EDT/R@1 across our two same-data runs (16×4 beats 8×8); the release used micro 64 — BN-stat quality in the RIR encoder + gradient-noise structure are micro-dependent. Testable.
- **P0 launched** — completing the selection curve: 10 unscreened ≥20k ckpts (K=8, seed 42, EMA), ~2 h. Gives the full 21-point band and the best-in-band candidate for a labeled 5-seed row.
- **P1 (proposed, needs approval)** — micro-parity B-V rerun: probe vanilla-only fit at 64×1 (else 32×2) → retrain 67.5k (~3.4 d). Removes the largest remaining controllable deviation. **P2 (conditional)** — if P1 still gaps on EDT/R@1: the reachable-factor list is exhausted → the residual is the authors' internal data version (unreachable) and/or DINOv3 snapshot; decision point.

## 2026-07-16T00:47—04:35 — B-V EXTEND launched (Yixun: extend-first→P1, adaptive 100k) + first point S70000: all metrics improve, R@1 = lineage max

*(By-line: launch + scripts authored by Opus 4.8 1M/max covering while Fable 5 was at its limit; analysis below by Fable 5 after Yixun switched back.)*

- **Approvals (2026-07-15/16):** Yixun approved P1 ("I approve P1"), then added the EXTEND experiment ("continue our previous train on B-V@67.5k to check what is the best ckpt we have"), ordered **extend-first then P1, adaptive to 100k** (screen 80k/90k/100k; continue toward 135k only if still improving). New standing directive: FLAC runs → wandb **yh4742@princeton.edu** — BLOCKED (current key = yixunhu21@gmail.com); runs stay `--logger none` behind a fail-closed identity gate until the key is swapped.
- **Launch (00:47, code `c40908c`):** `bv_extend_launch.sh 100000` — full-state resume of `epoch=14-step=67500.ckpt` (PL "Restored all states"; lr 4.84e-5 exact-correct for InverseLR@67.5k; loss 0.32–0.6; 10.4 GiB; PID 3737059). NOT bit-exact by design: PL 2.1 restores no RNG and does not fast-forward the dataloader mid-epoch (warning captured in log) — fresh stochastic continuation, fine for best-ckpt search. Review verdict on the launch script: **SHIP** (`p1_scripts_codex_reverify.md`).
- **Screen-driver incident (03:42, v1 inline):** `{...} | tee | tail -0` — `tail -n 0` exits without reading → SIGPIPE killed tee+block **before any eval started** (0-byte log, no metrics, training untouched). Replaced by `bvext_screen.sh` (direct `>>` redirect, fail-propagating; reviewed **SHIP**, `bvext_screen_codex_review.md`; committed `87e6cdb`).
- **S70000 (EMA, K=8 s42 full split):** T60 **9.107** / C50 **0.9368** / EDT **40.538** / FD 0.3112 / R@1 **6.486**. Deltas vs 67.5k endpoint: T60 −0.41, C50 −0.037, EDT −2.23, FD −0.009, R@1 +0.32 — **every metric improved in 2,500 steps**; C50 now BELOW released (0.9682); **R@1 6.49 > phase-1 curve max (6.22@65k) = best B-V R@1 in this lineage**. Online: 9.376/0.987/40.78/6.20 — still trails EMA. Caveats: single point inside a wide oscillation band (late-EDT band ≈38.3–42.8); T60/EDT values are band-consistent, only R@1 exceeds the band. 80k/90k/100k decide trend-vs-draw.

## 2026-07-17 19:25—21:30 — EXTEND COMPLETE (100k, rc=0) + full 13-pt curve: **R@1 REACHES RELEASED PARITY at 92.5k (7.054 vs 7.06)** — under-training CONFIRMED for R@1; EDT stays systematic

*(Interleaved ops, 2026-07-16 pm → 07-17: extend leg-1 stopped at 77.5k for Yixun's B-F DDP+SyncBN reprioritization [`bv_extend_stop_restart.md`]; leg-2 resumed 16:39 under wandb (`FLAC_exp07_BVextend/969vypp5`) and ran 77.5k→100k clean, rc=0 at 19:24:47. SyncBN `--sync-batchnorm` wiring landed TDD (40 tests, review+reverify SHIP, `f362673`); training env switched to conda `flac` per Yixun (flash-attn kernel delta disclosed; evals stay `rir2rir`); wandb key = yh4742 verified.)*

- **Full extend curve (EMA, K=8 s42, full split; endpoint anchor 67.5k = 9.516/0.9740/42.763/R6.170):** 70k 9.107/0.9368/40.538/6.486 · 72.5k 9.325/0.9712/40.547/6.359 · 75k 9.436/0.9688/**39.782**/6.470 · 77.5k 9.612/0.9716/41.479/6.596 · 80k 9.866/0.9540/41.472/6.738 · 82.5k 9.955/0.9631/41.712/6.707 · 85k 9.803/0.9834/40.792/6.833 · 87.5k 9.661/0.9872/42.365/6.486 · 90k 9.836/1.0110/42.359/6.817 · **92.5k 9.321/0.9776/40.706/7.054** · 95k 10.030/0.9990/43.946/6.517 · 97.5k 9.710/1.0119/43.392/6.438 · 100k 9.754/0.9729/43.381/6.722.
- **R@1: released parity REACHED.** 7.054@92.5k vs released 7.06 (Δ0.006 ≈ 0.06× eval σ 0.10). Climb 6.17→7.05 over 25k extra steps; peak-then-oscillate (95k–100k fall back to 6.4–6.7). **P0's R@1 gap is resolved as a BUDGET artifact, not lineage.** Held-out-seed confirm (43–46, K=8 EMA) launched 21:2x per the pre-registered candidate rule.
- **EDT: the sole surviving systematic gap.** Extend best 39.78@75k; late points degrade (43+ at 95k–100k); lineage best remains 38.29@60k vs released 37.10.
- **T60: never revisits the 8.3–8.6 zone** (mid-training 30–40k territory); late band 9.1–10.0 — late training trades T60 for R@1. **C50: at target** (oscillates around released). FD best 0.3077@72.5k.
- **Best-ckpt answer (Yixun's original ask):** no checkpoint dominates; the pre-registered composite rule (T60≤8.63 ∧ C50≤0.974 ∧ EDT≤37.27) matches NOTHING anywhere in the lineage (T60/EDT bounds never met). **Recommended single ckpt: `epoch=20-step=92500` — the only released-level-retrieval point in the entire B-V lineage, with C50 near-target and mid-band T60/EDT.** Metric-priority alternatives: EDT-best 60k (38.29, phase 1), T60-best 30k (8.34, phase 1).
## 2026-07-28 ~07:30 — **FULL TABLE-1 PARITY: ckpt 87,500 5-seed CONFIRMED at BOTH K — 8/8 cells SUPERIOR or EQUIVALENT. MAXIMUM PROJECT GOAL CLOSED.**

- **K=8 (5 seeds):** T60 **8.2930±0.0106** (−19.8σ_c SUPERIOR) · C50 **0.9660±0.0015** (−0.65σ_c EQUIV) · EDT **35.9513±0.0532** (−13.1σ_c SUPERIOR) · R@1 **6.9592±0.1353** (−0.60σ_c EQUIV).
- **K=1 (5 seeds):** T60 **9.5401±0.0231** (−9.5σ_c SUPERIOR) · C50 **1.0323±0.0060** (−1.6σ_c SUPERIOR) · EDT **38.7283±0.2263** (−2.8σ_c SUPERIOR) · R@1 **6.8108±0.1766** (−0.07σ_c EQUIV).
- **Checkpoint of record: `outputs_FLAC/exp07_P1/.../epoch=19-step=87500.ckpt`** (B-V vanilla, SyncBN-64 DDP recipe + ViT grad-ckpt, seed 42, 87.5k steps = 67.5k budget + 20k extension). 5 of 8 cells SUPERIOR (T60 both K by wide margins, EDT both K, C50 K=1); 3 EQUIV (≤1σ_c) incl. both R@1 — no cell worse than 1σ_c below released. **"Beat released Table-1 K=1/K=8": T60/EDT strictly beaten at both K, C50 beaten at K=1/equal at K=8, R@1 statistically indistinguishable.**
- Discovery ops note: the 87.5k screen's crossing was masked by a summary-plumbing bug in the batched screen block (metrics json was fine); caught on manual verification minutes later, confirm fired by hand. The chained-confirm plumbing bug is moot (no further crossings needed).
- Post-verdict curve points: 92.5k 8.994/**0.9300 (new program-best C50)**/37.761/R6.533 · 95k 8.488/0.9619/37.133/R6.407 — both below the 87.5k peak, reinforcing it as checkpoint of record. 97.5k/100k to land ~10:30, then the closing package — TODAY.

## 2026-07-27 — R@1 extension progress (leg 3 after a second teardown kill/resume): 70k R6.23 → 75k **R6.675** (crossing 6.9 nears)

- Ext screens (EMA s42): 70k 8.079(lineage-best T60)/0.9390/37.228/R6.233 · 75k 9.116/0.9407/39.575/R6.675 · 80k 8.804/**0.9332 (program-best C50)**/37.132/**R6.833** (0.07 below crossing) · 85k 8.906/0.9569/38.023/R6.249 (oscillation) · 90k 8.785/1.0099/**36.598**/**R6.849** — third near-miss at the 6.9 line (6.83/6.25/6.85); oscillation ceiling ≈ the crossing itself; 92.5k (the 8×8's exact parity step) next. Ops: teardown killed the extension twice (11:45 leg died ~15:35); re-resumed 16:32 from 72.5k (`mkum1n79`).

## 2026-07-27 ~01:30 — GATE VERDICT (5-seed, held-out 43–46): **57.5k SUPERIOR to released Table-1 on T60/C50/EDT at BOTH K — 6/8 cells SUPERIOR/EQUIV; R@1 the sole remaining gap**

- **57.5k K=8:** T60 8.4854±0.0071 (−8.9σ_c SUPERIOR) · C50 0.9636±0.0016 (−1.4σ_c SUPERIOR) · EDT 36.3789±0.1103 (−5.5σ_c SUPERIOR) · R@1 6.1607±0.0875 (−6.8σ_c OUT).
- **57.5k K=1:** T60 9.8793±0.0542 (−1.3σ_c SUPERIOR) · C50 1.0412±0.0061 (EQUIV) · EDT 39.2813±0.2387 (−1.5σ_c SUPERIOR) · R@1 5.8545±0.1929 (−3.3σ_c OUT).
- **67.5k endpoint (both K):** weaker — K=8 T60/C50 drift back OUT (+9.0/+2.1σ_c), EDT stays SUPERIOR (−1.5σ_c), R@1 OUT; K=1 NONINF/EQUIV×3 + R@1 OUT. **57.5k = the arm's checkpoint of record.**
- **Program-goal reading:** the maximum goal ("beat released Table-1 K=1/K=8") is achieved on 3 of 4 metrics at both K by a single seed-confirmed checkpoint; the EDT gap that survived P0, the 100k extend, and every prior arm is INVERTED (−0.72 K=8). **R@1 (−0.90/−0.98) is the sole open metric** — and P1's R@1 trajectory ran ahead of the 8×8's matched-step pace throughout (the 8×8 needed 92.5k for R@1 parity), making a budget extension of THIS recipe the natural closing move (Yixun's call; echoes the extend precedent).
- Next: closing package (results.md + analysis + HTML + tracker/issue/HANDOFF + closure review), then the strategic fork to Yixun.

## 2026-07-26 21:5x — P1 COMPLETE (67.5k, rc=0) + endpoint block phase 1: **EDT STRONG-threshold PASS (81% gap closure)** + **FIRST-EVER composite-rule qualifier (57.5k, all 3 error metrics sub-released)**

- **Full P1 curve (EMA K=8 s42):** 10k 11.78/1.378/47.48/1.78 · 20k 8.44/1.095/45.42/2.92 · 30k 9.20/1.096/43.43/4.17 · 40k 8.99/1.008/40.62/5.19 · 50k 8.65/0.985/37.65/5.54 · 55k 8.51/0.951/36.99/6.00 · 57.5k **8.493/0.9625/36.427**/6.06 · 60k 8.89/1.015/38.99/6.33 · 62.5k 8.82/0.949/38.16/5.87 · 65k 8.89/0.960/38.46/6.04 · 67.5k 8.77/0.973/36.95/6.28.
- **Late-curve statistic (mean 55k–67.5k):** T60 8.728 / **C50 0.9684 (= released 0.9682)** / **EDT 37.664 → STRONG PASS (≤38.59; 8×8 baseline 40.087; 81% of the released-gap closed)** / R@1 6.096 → threshold 6.51 FAIL (8×8 baseline 5.960; ~12% closure) ⇒ two-metric STRONG tier NOT met; EDT singly met, decisively.
- **COMPOSITE RULE: first qualifier in the program — step 57,500** (T60≤8.63 ∧ C50≤0.974 ∧ EDT≤37.27 was never met by any of the ~40 screened ckpts across B-V/extend/B-F): **8.493/0.9625/36.427 — all three error metrics BELOW released Table-1 simultaneously.** R@1 6.06 vs released 7.06 = the remaining shortfall (PARITY tier requires R@1 on held-out seeds; will not confirm at 6.06 — but the 3-metric sub-released claim goes to seed confirmation).
- **Phase 2 launched (overnight):** 18 evals — 57.5k and 67.5k × K∈{8,1} × 5 seeds → verdict + closing package in the morning.

## 2026-07-26 pm — P1@60k: R@1 6.328 (+0.57 over anchor at matched step); EDT band point 38.99

- P1 S60000 (EMA): 8.893/1.0146/38.991/0.3176/**R6.328** vs anchor 9.151/0.9679/38.294/0.3142/5.760. R@1 trajectory clearly faster than 8×8's (which needed 92.5k for 7.05). Endpoint block tonight: screen 55k/57.5k/62.5k/65k/67.5k (late-curve statistic inputs) + composite pick + 5-seed gate.

## 2026-07-26 — P1@40k/50k: **EDT 37.649@50k — best B-V-family EDT ever recorded** (8×8 lineage floor was 38.29); BN=64 hypothesis ALIVE

- P1 S40000 (EMA): 8.989/1.0076/40.620/0.3221/R5.192 (vs anchor 8.603/1.0047/39.991/0.3111/4.829 — R@1 ahead, rest ≈). P1 S50000: **8.647/0.9854/37.649**/0.3288/**R5.539** (vs anchor 8.983/0.9858/40.343/0.3221/5.255; released 8.609/0.9682/37.10/—/7.06).
- **50k = the closest simultaneous T60+C50+EDT to Table-1 in any of our runs**; EDT below the 8×8 lineage's all-time floor. Original P1/BN hypothesis (SyncBN-64 → EDT) revived; single band-point caveat — pre-registered late-curve statistic (55k–67.5k mean, thresholds EDT ≤38.59 / R@1 ≥6.51 for STRONG) + 5-seed endpoint gate adjudicate.
- Ops: resumed leg (lr45v31g) at ~0.5 steps/s exclusive; 40k+50k screens ran back-to-back post the batched monitor events.

## 2026-07-25 — P1@30k: R@1 watch-item RESOLVED (4.166 > anchor 4.040) — recipe clean on every metric

- P1 S30000 (EMA): T60 9.200 / C50 1.0958 / EDT 43.428 / FD 0.3392 / **R@1 4.166** vs anchor 8.338/1.0116/40.958/0.3174/4.040 (T60/EDT in-band, C50 near-equal, R@1 AHEAD). Attribution verdict stands caveat-free.

## 2026-07-24 (evening) — P1@20k CONFIRMS: attribution VERDICT — recipe innocent, fa-from-scratch guilty

- **P1 S20000 (EMA):** T60 **8.442** (< released 8.609!) / C50 1.0954 / EDT 45.418 / FD 0.3383 / R@1 2.919 — vs anchor 8.920/1.0941/45.616/0.3326/3.740, vs B-F 11.701/2.0927/84.712/0.3797/0.505.
- **Verdict (10k+20k concordant):** recipe effect ≈ 0 on T60/C50/EDT/FD (T60 better); R@1 −0.8 vs anchor (early-spread scale, WATCH at 30k/40k). Single-delta FA effect = the full crater (EDT +39, C50 +1.0, R@1 −2.4, step-time ×3.5). **fa_invariant-from-scratch starves its own conditioning; equivariance via FA = fine-tune-stage property (exp_08 corroborates).**
- P1 rides to endpoint (~Jul 26 night) for the micro/BN-parity + released-parity estimands — T60 already sub-released at 20k makes the SyncBN-64 endpoint genuinely live.

## 2026-07-24 — P1@10k: RECIPE INNOCENT — P1 ≈ the 8×8 anchor; B-F's gap attributes to fa-from-scratch itself

- **P1 S10000 (EMA, flac):** T60 11.784 / C50 1.3775 / **EDT 47.481** / FD 0.3836 / **R@1 1.783** — vs 8×8 anchor 9.817/1.298/48.685/0.373/1.909 (≈ equal; EDT slightly BETTER, T60 single-point lag in its oscillation band) and vs B-F-same-recipe 12.701/2.293/88.516/0.414/0.316 (chasm).
- **Sequential decomposition @10k:** recipe/environment effect ≈ 0 (P1−anchor: EDT −1.2, R@1 −0.13); total FA-arm effect ≈ the entire gap (BF−P1: EDT +41.0, R@1 −1.47; ~30× eval σ). Also attribution datapoint #1: P1 runs 3.5× faster than B-F at the same recipe (0.259 vs 0.074 steps/s — fa's ×4 ViT forward).
- **Reading:** fa_invariant-from-scratch is itself the crippling factor; coheres with exp_08 (fa fine-tuned from converged vanilla → exact C₄ + T60 gains). Route-1 narrative: equivariance via frame-averaging = fine-tune-stage property (or needs a from-scratch curriculum). 20k tonight formalizes; P1 rides to endpoint for the parity estimands regardless.

## 2026-07-22 — B-F 30k: slope FROZEN at ~2× (EDT) / 0.15× (R@1); futility package pre-staged for the 50k review

- **BF 30k (EMA):** T60 11.546 / C50 2.1107 / EDT 80.346 / FD 0.3753 / R@1 0.615 (online 11.743/2.148/81.4/0.694). vs BV@30k 8.338/1.0116/40.958/0.3174/4.040. 20k→30k gains (EDT −4.4, R@1 +0.11, C50 +0.02 WORSE) ≤ B-V's own same-window gains → **relative deficit no longer closing**; extrapolation to 67.5k ≈ EDT 65–70 / R@1 ~1.
- **Read:** convergence toward a much worse operating point, not a slow start (conditioning confirmed active by the cfg0 probe). Attribution recipe-vs-fa still requires P1 (B-V at the identical DDP+SyncBN+ckpt recipe).
- **Posture:** 50k futility review (~Jul 25) is the pre-registered decision point; options pre-staged to Yixun (ride to 67.5k / stop@50k→P1 / stop-now→P1); early decision offered, no unilateral action. aug291k ended Jul 20 (step 100k); B-F pace ~0.074–0.10 steps/s; two third-party ~6.5 GB co-tenants (not ours).

## 2026-07-20/21 — B-F early screens (10k/20k) + conditioning-lift probe: trajectory globally slow, conditioning ACTIVE; ride to 50k review

- **Screens (EMA K=8 s42; flac-env evals — env-bridge proved flac≡rir2rir to 4 decimals):** BF 10k: 12.70/2.293/88.5/FD .414/R .32 · BF 20k: 11.70/2.093/84.7/.380/.51 — vs BV anchors 10k 9.82/1.30/48.7/.373/1.91, 20k 8.92/1.09/45.6/.333/3.74. **B-F not closing the relative gap** (still ~1.9× EDT, R@1 ~0.14×), loss meanwhile healthy (0.44@e4) → escalation criterion fired.
- **D1 conditioning-lift probe (cfg-scale 0 vs 1, both 20k ckpts):** BF lift T60 −19.0 / C50 −2.09 / EDT −104.6 / FD −.100 / R@1 +0.33 vs BV −17.5 / −1.88 / −88.6 / −.085 / +2.23. **Conditioning is ACTIVE and lifts ≥ B-V absolutely on all error metrics** — dead-path/signal-dilution hypotheses REFUTED. B-F is uniformly behind (cfg0 AND cfg1): a globally slower trajectory, only R@1-grade specificity disproportionately late (cf. B-V needing 92.5k for R@1 parity). Attribution recipe-vs-fa impossible without P1 (the 8×8 BV was never the matched control — pre-registered).
- **Decision: continue per plan** (hard-aborts only; futility review at 50k ~Jul 25 with slope from 30k/40k screens; options there = extend-budget / P1-first-for-attribution / stop).

## 2026-07-18 16:30—17:45 — M1 RUNG REPORT: the BN=64 rung (32/GPU) OOMs on 48 GB even solo — mandate needs an intervention; options to Yixun

- **Policy change (Yixun):** co-tenancy allowed — M1 gate switched from zero-procs to per-GPU `memory.free ≥ 45,087 MiB` (38.53 GiB M0 allocation + 1.5 DDP/SyncBN + 4 margin, review-corrected units) with co-tenant disclosure; throughput flagged noisy; fit bit = signal (`a18e684`).
- **Arithmetic veto on 2-rank co-tenancy:** GPU 0 headroom 27.8 GB < even micro-8 B-F's measured 36.8 GB peak → no B-F rank co-resides with aug291k (which passed 67.5k → target unknown). → single-rank **fit proxy** on idle GPU 1 (micro 32, env flac/flash-DiT, no SyncBN — needs ≥2 ranks; reviewed, 4 findings fixed verbatim).
- **Proxy result 1 (plain allocator):** OOM in 28 s, peak 48,514 MiB — card exhausted.
- **Proxy result 2 (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`):** **STILL OOM** (26 s, 47.26 GiB in use) → fragmentation was NOT the whole story; **true micro-32 B-F per-rank demand ≥ 47 GB** (C₄ frame-averaging → ViT batch 128 dominates; flash-DiT irrelevant to the ViT path).
- **Grad-ckpt wiring saga (18:10–19:45):** round 1 wired HF `gradient_checkpointing_enable` (49 tests green). Codex review P0'd it as "a no-op in 4.57.0 — DINOv3's loop never invokes `_gradient_checkpointing_func`"; I file-local-grep-"confirmed" the same. **Round 2 REFUTED the P0 empirically:** the mechanism lives in the `GradientCheckpointingLayer` BASE class (`transformers/modeling_layers.py` `__call__`, visible in the M0 OOM stacks all along); an execution trace showed **12 real checkpoint segments** through the unmodified HF path — the enable-time `functools.partial` merely makes it shim-invisible. ⚠️ Correction: `gradckpt_codex_review.md`'s "no-op" claim must NOT be cited as fact. The explicit per-layer adapter was shipped anyway (strictly tighter: grad-enabled gating, call-site `use_reentrant=False`, structural fail-closed incl. a decorative-enable backbone test): **54/54 tests; execution proof 12/12; ON-vs-OFF param grads bitwise identical (max abs diff 0.0, 210 tensors); state_dict sha256 unchanged; arm gate re-passed (hash `44a2f6aa…`, BF key on disk).**
- **VERDICT: the BN=64-compliant rung cannot run on 48 GB A6000s as-is.** 2-rank M1 moot (watcher stopped). Options put to Yixun: (a) BN=32 compromise — 16/GPU×2×accum2+SyncBN (+alloc flag; micro-16 measured 37.8 GB allocated → likely fits; 10-min proxy to confirm); (b) **BN=64 via ViT gradient checkpointing** — micro-32 + HF `gradient_checkpointing_enable(use_reentrant=False)` on the DINOv3 conditioner: numerics-IDENTICAL (recompute-in-backward), kills the ViT activation peak (every OOM site is inside the ViT), ~15–25% slower, needs a small TDD-wired flag + review + proxy — the only option that delivers the full mandate; (c) revert to single-GPU 8×8 (BN=8, the pre-DDP plan). **HOLD maintained — no training until Yixun picks.**

- **92.5k held-out-seed confirm (22:15, K=8 EMA seeds 42–46):** R@1 **6.921 ± 0.186** vs released 7.06±0.10 → Δ −0.139 = **0.66σ_c (σ_c=√(0.186²+0.10²)=0.211) → within the ≤1σ_c EQUIVALENCE tier — R@1 parity CONFIRMED under the pre-registered 5-seed protocol** (single-seed 7.054 was mildly optimistic; 4/5 seeds ≥6.88, seed 44 low draw 6.61). T60 9.312±0.012 / C50 0.9785±0.0007 / EDT 40.696±0.052 (seed-stable; C50 ~1e-3 above the 0.974 band edge, same K=8 story as the endpoint gate).
