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
