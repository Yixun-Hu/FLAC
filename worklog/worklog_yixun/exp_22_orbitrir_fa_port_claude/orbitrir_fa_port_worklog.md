# exp_22 — ORBITRIR frame-averaging port — lab notebook

## 2026-08-29T12:45:00-04:00 — clone upstream master into ~/codespace/ORBITRIR
- **Goal** — Yixun query 1, part 1: a fresh checkout of `AmandineBtto/FLAC` `master`.
- **Command / Validation** — `git clone -b master https://github.com/AmandineBtto/FLAC.git ~/codespace/ORBITRIR`; `git log --oneline -1` → `ead8bbd Add ArXiv link` (4 commits total: 2e3f847 → 0426d85 → 9eaa3e2 → ead8bbd). Tree: AGREE/ assets/ baselines/ data/ src/ + train.py eval_FLAC.py eval_pl.py eval_VAE.py unwrap_model.py defaults.ini pyproject.toml download_weights.sh README.md.
- **Result** — `passed`. Remote `origin` = AmandineBtto (read-only for us); git identity Yixun-Hu / yixunhu21@gmail.com inherited from global config.
- **Analysis** — ancestry check: `git merge-base 0bd5da0 ead8bbd` = `ead8bbd`, i.e. the FLAC fork's SOP base `0bd5da0` is upstream master + local setup commits (relative dataset/HF/AGREE paths, in-tree AGREE import in `metric_callback.py`, `device_map="auto"` removed in `conditioners.py`, exp_02 `yaw_rotation.py` + `eval_FLAC.py --rotate-deg`). So every fork change is expressible as a linear delta on ORBITRIR's base.
- **Next** — scope the B-F delta (upstream → B-F launch commit `f59f5a4`), write the plan, get approval.

## 2026-08-29T12:55:00-04:00 — scoping: what B-F actually needs from upstream→f59f5a4
- **Goal** — separate the B-F-essential code from the other experiments' code in the 6,225-line upstream→`f59f5a4` delta.
- **Command / Validation** — `git diff --stat ead8bbd f59f5a4 -- src/ train.py eval_FLAC.py defaults.ini`; per-file hunks; `git log ead8bbd..f59f5a4 -- src/ train.py eval_FLAC.py`; `git log f59f5a4..HEAD -- src/data/yaw_rotation.py src/training/diffusion.py src/models/conditioners.py` (post-launch changes to the FA path).
- **Result** — `passed`. B-F-essential (as-run at `f59f5a4`): `src/data/yaw_rotation.py` (exp_02 rotation primitives + exp_03 `wrap_angle`/`cylindrical_pose_features`/`DEFAULT_FRAME_ANGLES`/`invariant_conditioning`/`rotate_scene_metadata(pose_keys=)`), `src/models/conditioners.py` (`MultiConditioner.forward(only_ids=)`, ViT gradient checkpointing, `device_map` removal), `src/training/diffusion.py` (`cond_method`/`frame_avg_angles` ctor + `_compute_conditioning` at 3 sites), `src/training/factory.py` (2 lines), `eval_FLAC.py` (`--rotate-deg`, `--cond-method`, `--frame-avg-angles`, `--cond-autocast`, `--allow-partial-load`, output-path/record/sidecar helpers, load-integrity check), `train.py` + `defaults.ini` (`--max-steps`, `--sync-batchnorm`), `src/metrics/metric_callback.py` (in-tree AGREE import), config `FLAC_AR_BF.json` (= `FLAC_AR.json` + 3 keys). Tests: conftest, test_yaw_symmetry, test_invariant_conditioning, test_cond_dispatch, test_eval_paths (needs decoupling from the exp_02 worklog comparator), test_train_max_steps, test_train_sync_batchnorm, test_vit_gradient_checkpointing (needs decoupling from the exp_07 worklog canary).
  NOT B-F: `finetune_cond.py` + test (exp_03 fine-tune driver; exp_04 warmup / exp_05 freeze-BN / exp_06 lr-schedule live inside it), `src/models/cyl_vit.py` + CylViT/SimpleViT config branches + `token_pool` + the two `self.log(...)` lines in diffusion.py (all exp_05_cylvit `50cd944`), `tools/bn_drift_probe.py`, the CylViT/SimpleViT model configs.
  Post-launch FA-path commits (exp_11 batched orbit `1479304`/`8094d60`, exp_14 `max_fwd_samples`, exp_15 yaw_aug, exp_16 ARE, exp_19 cyl routing, exp_21 fa_cartesian) are all ADDITIONS, none a bug fix to the as-run B-F path → port the exp_07-era (loop-over-angles) version.
- **Next** — plan written (`plan_orbitrir_fa_port.md`); awaiting Yixun's decisions D1–D7 + approval before any code is written in ORBITRIR.

## 2026-08-29T13:15:00-04:00 — D2 resolved: ORBITRIR remote + branch set by Yixun
- **Goal** — record the target repo/branch decision.
- **Result** — `passed`. Yixun re-pointed `origin` to `git@github.com:Yixun-Hu/ORBITRIR.git`, kept `upstream` = `https://github.com/AmandineBtto/FLAC.git`, renamed the local branch to `main` (tracks `origin/main`) and pushed the base: `git ls-remote origin` → `refs/heads/main` = `ead8bbd`. Verified by this session; nothing else changed in the tree.
- **Analysis** — D2 closed; every port commit lands on `origin/main`. D1, D3–D7 + plan approval still open.
- **Next** — on Yixun's go: Coder round R1 (C1–C3).

## 2026-08-29T13:40:00-04:00 — plan Rev 2: re-sliced to ≈200 lines per commit (Yixun's readability request)
- **Goal** — Yixun asked whether every migration slice can be ≈200 LoC. Measured the real sizes instead of estimating.
- **Command / Validation** — `git diff --numstat` per planned hunk (`ead8bbd`/`0bd5da0`/`e85ebde` → `f59f5a4`), `ast`-based per-function line counts for `yaw_rotation.py`, `eval_FLAC.py` helpers and all 7 test files; `git grep yaw_transform_consistency f59f5a4` (only user: archived exp_02 diagnostic).
- **Result** — `passed`. Code hunks are small (max: eval wiring 223, grad-ckpt 73, invariant_conditioning 87) — the test files are what inflate a commit (194–513 lines). Rev 2: 14 commits, 70–240 lines each (median ≈210): old C3 folded into C2; C5/C7/C10 split into code+core-tests / remaining-tests; fork-development guard tests trimmed (new D8, ≈250 lines); `yaw_transform_consistency` dropped (new D9).
- **Next** — awaiting Yixun's go on D1, D3–D9.

## 2026-08-29T14:30:00-04:00 — Codex plan review (via Yixun) verified point by point → plan Rev 3
- **Goal** — Yixun pasted a Codex REQUEST-CHANGES review of plan Rev 2 and asked whether it is correct.
- **Command / Validation** — each finding checked against the repo (not taken on trust): `git show f59f5a4:src/tests/test_vit_gradient_checkpointing.py` (worklog `FLAC_AR_BV.json` import at lines 77-78/101; `allclose(atol=1e-6, rtol=1e-5)` contract), `test_eval_paths.py` (`_FakeEvalModule` + empty dataloader), `test_yaw_symmetry.py` (`rotate_pose_keys_default` asserts only `depth` changed), `git diff ead8bbd 0bd5da0 -- AGREE/AGREE/model_configs/dinoV3.json` (VAE path), `ORBITRIR/src/inference/generation.py:51,56` (raw `model.conditioner`), `assert_arm_configs.py:56` (`VIT_REV`), `conditioners.py` (`from_pretrained` without `revision`), `HEAD eval_FLAC.py` (fa_cartesian-only binding; fa_invariant "recorded, never enforced"), torch.load probe of the B-F 40k ckpt (embedded `model_config`: `fa_invariant`, angles, grad-ckpt ×2) and of `weights/FLAC/FLAC_EMA.ckpt` (bare `state_dict`), `sha256sum` of the B-F 40k ckpt, the s42 K=8/K=1 registered JSONs.
- **Result** — `passed`: **all seven findings CORRECT** (and Codex's quoted 8.190/0.9804/38.811/5.302 is exactly the K=8 s42 JSON). Saved verbatim + verification table in `orbitrir_fa_port_codex_plan_review.md`. Plan Rev 3 written: +C7c trained-as guard (inherit-by-default, fail-closed mismatch, `--allow-conditioning-override`, trained/runtime recorded, `eval_pl.py` too); C1 = explicit 19-file inventory incl. `dinoV3.json` + `ViT.revision` loader support; C2 physical rotation-sign/`alpha_eff`/co-rotation/non-mutation tests; C6 shared `dispatch_conditioning()`; C7b one-batch spy test; C10b in-tree `FLAC_AR.json` + in-memory flag, `allclose` contract, tests must RUN; C11 four type-strict leaf paths, no DINO build; §3.5 acceptance pinned (sha256 `5319feb4…2328`, s42 K=8/K=1 expected values, tolerances, negative control with the guard); C12 supported-entry-points section. 15 commits. New decisions D10 (pin `FLAC_AR.json` too), D11 (shared dispatcher), D12 (two-cell vs full matrix).
- **Analysis** — the two findings with the largest consequence are 1 (the guard turns the announcement-05 footgun into a `ValueError`) and 2a (without it every DINO-backed test would have silently skipped in the clean repo — a green suite proving nothing).
- **Next** — awaiting Yixun's go on D1, D3–D12.

## 2026-08-29T14:55:00-04:00 — plan APPROVED (Rev 3, all recommendations); Coder round C1 launched
- **Goal** — C1 "Portability fixes + pinnable HF revision" in `~/codespace/ORBITRIR` (Yixun gate: present C1, start C2 only after his approval).
- **Change (spec handed to the Coder, Opus 5 max effort)** — (a) 19 config files → repo-root-relative paths, byte-identical to their `0bd5da0` versions incl. `AGREE/AGREE/model_configs/dinoV3.json`; (b) `src/metrics/metric_callback.py` in-tree `AGREE.AGREE` import; (c) `src/models/conditioners.py` drop `device_map="auto"`, optional `ViT.revision` → `from_pretrained(..., revision=)` (absent → no kwargs); (d) README path/flag fixes only (`weights/FLAC/…`, `weights/AGREE/…`, real argparse flag names, trailing-space continuation); (e) `.gitignore` runtime dirs; (f) `src/tests/conftest.py` (moved here from C2 — without the repo-root `sys.path` pin the C1 tests would import a stale pip-installed `src`); (g) `src/tests/test_portability.py`.
- **Acceptance criteria (before verdict)** — `py_compile` clean; `pytest src/tests/test_portability.py` all PASSED in `flac`; the 19 parity diffs vs `0bd5da0` empty; `git diff --stat` ≈ 170 lines (projection; measured value recorded in `commits_*.md`); no `exp_NN`/worklog/reviewer references in any added text; no commit/push by the Coder (main session commits after the ladder).
- **Result** — `in_progress` (Coder running).
- **Next** — ladder → local commit on `main` (not pushed) → present C1 to Yixun → on approval C2 → R1 Codex review (C1+C2) → fixes → push.

## 2026-08-31T10:22:23-04:00 — C1 COMMITTED (`7d7d185` on ORBITRIR main, unpushed)
- **Goal** — close the C1 Coder round per Yixun's instruction ("use <his 4-line revision comment> … And then commit").
- **Change** — Coder output verified independently (py_compile OK; pytest 4/4 in `flac`; 19/19 config parity vs `0bd5da0`; no exp/worklog refs; mutation test: each production change reverted → exactly 1 test fails). Then the `revision` comment in `src/models/conditioners.py` replaced with Yixun's exact wording (fail-closed unique-match replacement), ladder re-run green, committed.
- **Version Control** — ORBITRIR `main`: `ead8bbd` → **`7d7d185`**, 25 files +179/−62. Not pushed; plan §3.2 pushes after review round R1 (C1+C2) closes.
- **Result** — `passed`.
- **Analysis** — Coder deviations accepted: dead `import os/sys` removed in metric_callback (spec-permitted); `eval_pl.py`'s `--store-predictions` left as-is (prefigure really defines the hyphenated flag; its inert `False` value is an upstream defect deferred to C7c). OPEN items deferred to the R1 review round (Yixun did not adopt them at commit time): (i) `.gitignore` patterns unanchored — `HAA/` also matches `data/HAA/` (new files there would be silently ignored) and no final newline; (ii) README `--output-dir` → `--out-dir` (baselines) + remaining trailing-space/missing-backslash lines (103–106, 195–201, 275–276, 299–300).
- **Next** — Yixun gate: his approval of C1 starts C2 (rotation primitives + `--rotate-deg` + physical sign tests). Then R1 Codex review of C1+C2 → fixes → push to `origin/main`.

## 2026-08-31T10:23:54-04:00 — C1 approved by Yixun (review-and-commit order) → Coder round C2 launched
- **Goal** — C2 "Yaw-rotation primitives + --rotate-deg diagnostic": new `src/data/yaw_rotation.py` (POSE_KEYS, azimuth_rotation_matrix, rotate_scene_metadata with pose_keys; yaw_transform_consistency DROPPED per D9), eval_FLAC.py --rotate-deg hunks (byte parity with `0bd5da0:eval_FLAC.py`), `src/tests/test_yaw_symmetry.py` part 1 with the Codex-mandated physical tests (panorama-consistency at 90/180/270/37.3°, wrong-sign negative control, alpha_eff quantisation dj=53@37.3°/dj=128@90°, pose/depth co-rotation, non-mutation).
- **Acceptance criteria** — py_compile clean; FULL suite green in `flac` (portability tests must still pass); `diff` vs `0bd5da0:eval_FLAC.py` EMPTY; `diff` vs `f59f5a4:src/data/yaw_rotation.py` shows only the enumerated omissions (no FA functions yet, no yaw_transform_consistency, exp-ref rewords); mutation check (roll-sign flip) fails the consistency test; ≈260 diff lines; no exp/worklog refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → present C2 diff to Yixun → commit → R1 Codex review (C1 `7d7d185` + C2) → fixes → push to origin/main. Deferred-to-R1 items standing: .gitignore anchoring/newline, README --out-dir + remaining continuation-line fixes.

## 2026-08-31T10:31:42-04:00 — C2 Coder round returned; independent ladder PASSED; diff presented to Yixun
- **Goal** — verify the C2 Coder output before presenting (the subagent's report carried an automated classifier security flag → full tree inspection performed, not just the report).
- **Command / Validation** — git status (exactly M eval_FLAC.py + 2 new files, nothing else); py_compile OK; full suite 11/11 in `flac`; eval_FLAC.py BYTE-IDENTICAL to `0bd5da0:eval_FLAC.py`; yaw_rotation.py vs `f59f5a4`: 265 reference lines dropped (DEFAULT_FRAME_ANGLES, wrap_angle, cylindrical_pose_features, invariant_conditioning, yaw_transform_consistency — all later-commit/D9 surface), only 4 reworded lines added (docstring drops the nonexistent standalone-diagnostic mention + exp-ref); banned-string scan clean; no subprocess/network/exec patterns in the new files. Coder's own mutation check: roll-sign flip → 4/7 tests fail (π deviation at 90°), restore verified by sha256.
- **Result** — `passed`; diff (19-line eval hunk + 99-line module + 226-line test file) presented to Yixun for the C2 commit approval.
- **Analysis** — Coder judgement calls accepted: docstring reword (the standalone diagnostic script doesn't exist in ORBITRIR); allclose(atol=1e-6) instead of torch.equal in the quantisation test; 3 strengthening asserts. Inherited caveat flagged for C7a: `rot_suffix` int-truncation collides 37.3° with 37.9° (verbatim from the reference; kept for parity — the registered naming scheme).
- **Next** — Yixun approves → commit C2 → launch R1 Codex review (C1 `7d7d185` + C2) with the deferred items (.gitignore anchoring/newline, README --out-dir + continuation lines) in the briefing.

## 2026-08-31T11:32:25-04:00 — C2 COMMITTED (`dd43998`) on Yixun's approval; R1 Codex review LAUNCHED
- **Goal** — close the C2 Coder round and open review round R1 (C1 `7d7d185` + C2 `dd43998`).
- **Version Control** — ORBITRIR main: `7d7d185` → `dd43998` (3 files, +342/−2). Still NOT pushed; push happens when R1 closes with fixes applied.
- **Command / Validation** — review: `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message orbitrir_fa_port_codex_code_r1_review.md "<briefed prompt>" < /dev/null`, backgrounded, full transcript teed to `orbitrir_fa_port_<ts>_r1_codex_review.log` (prompt recorded at the top). Briefing: SOP + announcements + plan Rev 3 + worklog + plan review; scope = ead8bbd..dd43998 only; C4-C12 absence explicitly out of scope; parity claims to verify; the four deferred items (gitignore anchoring/newline, README --out-dir + continuation lines, rot_suffix truncation, eval_pl inert flag) to adjudicate fix-now vs defer; installs/env-writes forbidden; SOP identity header required.
- **Acceptance criteria** — review file lands with the identity header + verdict; BLOCKING/Major findings get fix commits before push; deferred-item adjudications recorded; round closes per SOP (review → fixes → re-verify → log) before C4 starts.
- **Result** — `launched`.
- **Next** — on review return: triage findings → fix commits (Coder) → re-verify → push `origin/main` → present R1 closure + C4 go-request to Yixun.

## 2026-08-31T11:39:33-04:00 — R1 review returned (REQUEST-CHANGES) → fixes applied → round CLOSED → pushed
- **Goal** — close review round R1 per SOP (review → fix → re-verify → log).
- **Result** — `passed`. Verdict REQUEST-CHANGES with 3 findings (1 Major: unanchored .gitignore; 1 Minor: README continuations + --out-dir + inert eval_pl example; 1 Nit: trailing whitespace on changed config lines) — all fixed in `28d0787`. Deferred-item adjudications: (i)+(ii) fix now (done), (iii) rot_suffix truncation → C7a with collision tests, (iv) eval_pl boolean semantics → C7c. Reviewer independently verified both parity claims (cmp/diff commands recorded in the review) and found NO implementation defect in C2's rotation/quantisation/co-rotation/non-mutation. Re-verify: git diff --check ead8bbd..HEAD clean; check-ignore proves data/HAA/ unshadowed + root HAA/ still ignored; suite 11/11.
- **Version Control** — ORBITRIR main `ead8bbd → 7d7d185 → dd43998 → 28d0787`, pushed to origin/main.
- **Next** — Yixun gate: go-request for round R2 (C4 cylindrical pose invariants → C5a/C5b invariant_conditioning → C6 dispatch).

## 2026-08-31T11:49:45-04:00 — gating decision: per-commit approval (Yixun "(b)"); Coder round C4 launched
- **Goal** — C4 "Cylindrical pose invariants": DEFAULT_FRAME_ANGLES + wrap_angle + cylindrical_pose_features into yaw_rotation.py (verbatim from f59f5a4, positioned to converge the file), 9 reference tests merged into test_yaw_symmetry.py.
- **Acceptance criteria** — py_compile clean; suite 20/20 in `flac` (4 portability + 16 yaw); `git diff --check` clean; module parity diff vs f59f5a4 = ONLY invariant_conditioning + yaw_transform_consistency omitted + the two approved docstring rewords; wrap_angle mutation fails ≥1 test then restores clean; no exp/worklog refs; no commit by the Coder; ≈240 diff lines.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → present C4 diff → Yixun approval → commit → C5a (same gate) → … → R2 Codex review after C6.

## 2026-08-31T16:10:18-04:00 — C4 COMMITTED (`03e7596`) on Yixun's approval
- **Result** — `passed`. Independent ladder pre-commit: suite 20/20, git diff --check clean, parity = 3 enumerated hunks, banned-string scan clean. Coder note recorded: test_cylindrical_values is load-bearing (a uniform wrap_angle sign flip passes the invariance test, only the value test catches it). Cn-orbit question from Yixun answered: frame_avg_angles is the extension point; C8/16/32 exact for W=512; cost linear in n (chunking = post-port option); optional cn_frame_angles helper deferred.
- **Next** — Coder round C5a (invariant_conditioning + MultiConditioner only_ids), diff to Yixun before commit.

## 2026-08-31T16:10:55-04:00 — Coder round C5a launched
- **Goal** — C5a: `invariant_conditioning` verbatim from f59f5a4 (loop-over-angles, as-run; ONE docstring reword dropping the plan-section reference), `MultiConditioner.forward(only_ids=)` hunk, new `test_invariant_conditioning.py` part 1 (fakes VERBATIM — byte fidelity over the plan's slimming estimate — + test_multiconditioner_only_ids + test_c4_exact_invariance).
- **Acceptance criteria** — py_compile clean; suite 22/22; git diff --check clean; yaw_rotation parity vs f59f5a4 = 3 approved rewords + yaw_transform_consistency omission ONLY; conditioners parity = C1 hunk + later-commit absences only; averaging mutation fails a test then restores clean; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C5a diff to Yixun → approval → commit → C5b (extended tests).

## 2026-08-31T22:34:31-04:00 — C5a COMMITTED (`d87063f`) with Yixun's amendments
- **Result** — `passed`. Yixun approved with: (1) test_average_correctness pulled forward (divisor mutation /(n-1) now FAILS it — closed the C5a coverage gap), (2) Route-1 reworded in both files, (3) helper strips kept; plus his requested call-count regressions — reference test_single_pass_nonvit pulled forward verbatim (non-ViT calls==1, ViT calls==n_angles, BN num_batches_tracked==1) + new test_call_counts_follow_angles binding to len(angles) via a (0,90) orbit. Ladder: 25/25; git diff --check clean; yaw_rotation parity = 5 enumerated hunks (module now reads "Compute yaw-symmetrized FLAC conditioning."); Route-1 absent repo-wide.
- **Next** — Coder round C5b: remaining reference tests (pose_entries_any_angle, deep_nonmutating, no_depth_single_pass, angles_first_must_be_zero, stale-depth negative pair). average_correctness/single_pass_nonvit are DONE — C5b must not re-port them.

## 2026-08-31T22:35:00-04:00 — Coder round C5b launched (tests-only)
- **Goal** — port the 5 remaining reference tests + stale-depth helper into test_invariant_conditioning.py (pose_entries_any_angle, deep_nonmutating w/ import copy, no_depth_single_pass, angles_first_must_be_zero, stale-depth negative pair); average_correctness/single_pass_nonvit already landed in C5a and must not be re-ported.
- **Acceptance criteria** — suite 30/30 in `flac`; git diff --check clean; stale-depth mutation fails a test then restores byte-exact (yaw_rotation parity back to the 5 enumerated hunks); banned-ref rewords reported; tests-only diff; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C5b diff to Yixun → approval → commit → C6 (dispatch) → R2 Codex review → push.

## 2026-08-31T22:52:18-04:00 — C5b COMMITTED (`ed4d4a6`) with the empty-tuple arm
- **Result** — `passed`. Suite 30/30 after the arm; whitespace clean. **Correction from Yixun recorded:** test_no_depth_single_pass is "no-depth/no-ViT → one base pass" — it builds `_build_cond(with_geometry=False)` and asserts output == plain single pass; it does NOT assert ViT calls == 1 (my C5b-launch prompt/summary had described it wrongly; the ported test itself was always verbatim-correct). Commit message uses his wording.
- **Next** — Coder round C6 (dispatch_conditioning + wrapper/factory wiring) → R2 Codex review → push.

## 2026-08-31T22:53:18-04:00 — Coder round C6 launched (last commit before R2)
- **Goal** — C6: shared `dispatch_conditioning()` in yaw_rotation.py (D11 — the one approved design deviation from the reference: reusable module function instead of wrapper-inlined branches, so C7b's eval can share it), wrapper ctor cond_method/frame_avg_angles + validation, `_compute_conditioning` delegation at all three step sites, factory +2 lines, test_cond_dispatch.py (6 reference tests, spy targets adapted to the dispatcher route) + ~30-line dispatcher unit-test section.
- **Acceptance criteria** — suite ≈39 green in `flac`; git diff --check clean; diffusion.py parity vs f59f5a4 = delegation body + reworded docstrings (exp_03/Route-1/flow_source parentheticals dropped) + later-work absence ONLY; factory parity = the 2 lines; upstream-relative diff shows exactly the intended additions; hardcoded-vanilla mutation fails a dispatch test then restores byte-exact; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C6 diff to Yixun → approval → commit → R2 Codex review (C4, C5a, C5b, C6) → fixes → push origin/main.

## 2026-08-31T23:26:19-04:00 — C6 COMMITTED (`d690e38`) on Yixun's approval; R2 Codex review LAUNCHED
- **Version Control** — ORBITRIR main: 28d0787 → 03e7596 (C4) → d87063f (C5a) → ed4d4a6 (C5b) → **d690e38 (C6)**. Unpushed; push when R2 closes.
- **Command / Validation** — codex exec gpt-5.6-sol xhigh, backgrounded, transcript → orbitrir_fa_port_<ts>_r2_codex_review.log, verdict → orbitrir_fa_port_codex_code_r2_review.md. Briefing: R1 closed context; scope 28d0787..d690e38; enumerated deviations per file (incl. D11 dispatcher + Yixun's test additions as intentional); adversarial focus on fakes-based test blind spots + dispatcher design; C7+ out of scope; installs/writes forbidden.
- **Acceptance criteria** — review lands with identity header + verdict; BLOCKING/Major fixed before push; round closes per SOP; then push origin/main and present R2 closure + C7a go-request to Yixun.
- **Result** — `launched`.

## 2026-08-31T23:35:44-04:00 — R2 review returned (REQUEST-CHANGES, 3 Major) → fix round launched
- **Result of review** — production parity CLEAN (factory byte-exact; only_ids exact; sync_dist absences traced to excluded 50cd944 by the reviewer independently). Findings, all test/validation strength: (1) frame_avg_angles unvalidated at construction + method-set duplication ctor/dispatcher; (2) all-three-sites test spies below the dispatcher — re-inlining the branch would pass; (3) only custom-orbit test uses a prefix orbit (0,90) + call counts only — angle VALUES unproven. All three verified correct by the Planner on inspection.
- **Change (fix round, Opus 5 Coder)** — Fix 1: shared resolve_conditioning_options() (VALID_COND_METHODS single source; fa_invariant angle rules at construction; dispatcher uses it; ctor block replaced). Fix 2: dispatcher-boundary spy test (delegating recorder on src.training.diffusion.dispatch_conditioning; both methods × 3 sites; exact args incl. override tuple (0,180)); construction/dispatcher validation tests. Fix 3: test_custom_orbit_values_control_average with the non-prefix orbit (0,180) vs independent two-frame mean + difference from the C4 orbit.
- **Acceptance criteria** — suite green (expect ≈48-52); three mutation checks each red→restored (branch re-inlined → boundary test fails; angle-values ignored → orbit-value test fails; validation removed → construction test fails); git diff --check clean; fix diff presented to Yixun BEFORE commit (per-commit gate (b)); then commit + push closes R2.
- **Result** — `in_progress`.

## 2026-09-01T20:20:25-04:00 — R2 fixes COMMITTED (`bffe709`) on Yixun's approval → round R2 CLOSED → pushed
- **Version Control** — ORBITRIR main 28d0787 → 03e7596 → d87063f → ed4d4a6 → d690e38 → `bffe709`, all pushed to origin/main. Suite 60/60; window whitespace check clean.
- **Analysis** — R2 loop complete per SOP (review → fix → re-verify → log). The training-side FA method is now fully ported and pushed: rotation primitives, cylindrical invariants, orbit averaging, config dispatch, shared validation. Remaining: eval side (C7a/b/c → R3), training flags + config + README (C8–C12 → R4), smoke + pinned two-cell acceptance.
- **Next** — Coder round C7a (eval helper functions; includes the R1-adjudicated collision-free rot-suffix with legacy-compatible integer names); diff to Yixun before commit.

## 2026-09-01T20:21:20-04:00 — Coder round C7a launched (eval helpers; first commit of the R3 window)
- **Goal** — five pure helpers + LOAD_WHITELIST_PREFIXES into eval_FLAC.py (additive; evaluate_model untouched until C7b) + test_eval_paths.py part 1 (13 ported helper tests + new test_rot_suffix_collision_free). Mandated deviation (R1 adjudication iii): rot suffix via format(deg,'g') — legacy-identical for integer angles (_rot90), distinct for fractional (_rot37.3 vs _rot37.9).
- **Acceptance criteria** — suite ≈74 green; git diff --check clean; eval_FLAC parity vs f59f5a4 = helper rewords + 'g' line + absent C7b wiring ONLY; int-truncation mutation fails the collision test while legacy tests stay green; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C7a diff to Yixun → approval → commit → C7b (evaluate_model wiring + CLI) → C7c (trained-as guard) → R3 review → push.

## 2026-09-01T20:42:42-04:00 — C7a COMMITTED (`ed114ac`) on Yixun's approval; Coder round C7b launched
- **Goal** — C7b: evaluate_model wired onto the helpers + shared conditioning stack (DEVIATION 1: resolve_conditioning_options replaces the reference's inline validation — also enforces fa angle rules at eval; DEVIATION 2: dispatch_conditioning replaces the inline fa/vanilla branch, per D11), cond_autocast_ctx, load-integrity call, four CLI flags, yaw_rotation module-docstring accuracy line; tests: raise-before-model-construction ×2, parser choices, one-batch spy test (rotation→dispatch order, rotated metadata reaches the dispatcher, record/name assertions, vanilla arm).
- **Acceptance criteria** — suite ≈80+ green; git diff --check clean; eval_FLAC parity vs f59f5a4 = prior rewords + 'g' + the two declared deviations ONLY; --help shows all five flags; order/stale-metadata mutation fails the one-batch test then restores byte-exact; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C7b diff to Yixun → approval → commit → C7c (trained-as guard, --cond-method default becomes inherit) → R3 review → push.

## 2026-09-01T21:37:15-04:00 — D13 (Yixun): --allow-partial-load DROPPED from the port
- **Goal** — apply Yixun's decision: delete the escape hatch now, re-add later only if ORBITRIR needs it. Context established first: the flag entered the fork in 992fe49 (exp_03 full-review launch condition), was used exactly once ever (exp_03's non-EMA FLAC.ckpt format diagnostic), and NO B-F/registered run needed it (wrapper leftovers are whitelisted; clean loads enforced).
- **Change** — folded into the uncommitted C7b tree (C7a stays append-only): check_load_integrity loses the kwarg + warning branch (always raises; message no longer names a nonexistent flag), evaluate_model loses the param, CLI flag removed, escape-hatch test reworked to test_load_integrity_stray_and_missing_are_fatal, parser test adjusted. Suite re-run green; --help shows no flag.
- **Result** — `passed`. C7c's "record allow_partial_load in the metrics JSON" improvement is now moot.

## 2026-09-02T09:37:03-04:00 — C7b COMMITTED (`23cb7f4`) on Yixun's approval; Coder round C7c launched (last of the R3 window)
- **Goal** — C7c: trained_conditioning() (embedded model_config → trained method/angles, bare ckpt → --model-config fallback), inherit-by-default --cond-method/--frame-avg-angles (None defaults), fail-closed mismatch unless --allow-conditioning-override, trained/runtime/source/override fields in metrics record + predictions meta, strict guard + record fields in eval_pl.py (no override there), float coercion in resolve_conditioning_options (approved call 2).
- **Acceptance criteria** — suite ≈95 green; guard fires before create_model_from_config (sentinel-tested); mismatch raises both directions + angle-order; override recorded; 3 mutation checks red→restored; git diff --check clean; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C7c diff to Yixun → approval → commit → R3 Codex review (C7a+C7b+C7c) → fixes → push origin/main → C8.

## 2026-09-03T10:31:49-04:00 — C7c COMMITTED (`becaef7`) on Yixun's approval; R3 Codex review LAUNCHED
- **Version Control** — ORBITRIR main: bffe709 → ed114ac (C7a) → 23cb7f4 (C7b) → **becaef7 (C7c)**. Unpushed; push when R3 closes.
- **Command / Validation** — codex exec gpt-5.6-sol xhigh, backgrounded, transcript → orbitrir_fa_port_<ts>_r3_codex_review.log, verdict → orbitrir_fa_port_codex_code_r3_review.md. Briefing: R1/R2 closed context; scope bffe709..becaef7; enumerated deviations (C7a 'g' suffix, D11 deviations 1/2, D13 removal, C7c new-design surface); adversarial focus (a)-(f) incl. guard-fooling, D13 always-raise, 'g' edge cases, one-batch fake blind spots, eval_pl coupling, R4 surprises.
- **Result** — `launched`.
- **Next** — on verdict: triage → fix round if needed (diff to Yixun before commit, per gate (b)) → push origin/main → C8.

## 2026-09-03T10:45:14-04:00 — R3 review returned (REQUEST-CHANGES: 1 BLOCKING + 3 Major + 1 Minor + 1 Nit) → fix round launched
- **Findings (all verified valid by the Planner):** (1) BLOCKING malformed embedded config falls back to the external JSON — spoofable authority; angles accept strings/bools/NaN; (2) Major partial/absent EMA overlay hybridises silently past check_load_integrity; (3) Major extraneous angles under vanilla mis-flagged as conflict (override disclosure polluted); (4) Major 'g' token: 6-sig-digit collisions + same-cardinality reordered orbits share a path — replaced by int-special-case + repr round-trip + orbit digest for non-default orbits (legacy names pinned unchanged); (5) Minor eval_pl record missing trained_frame_avg_angles/conditioning_override; (6) Nit stale dispatcher docstring. Reviewer's adversarial pass otherwise CLEAN (guard sentinel flow, D13 always-raise, eval_pl coupling, one-batch fake disclosed blind spots as known).
- **Per Yixun's standing instruction** ("if R3 is clean, push and go ahead with C8"): NOT clean → push held; fix round (Opus 5 Coder) launched with the six prescriptions + 4 mutation checks; fix diff goes to Yixun before commit (gate (b)).
- **Result** — `in_progress`.

## 2026-09-03T10:50:04-04:00 — R3-fix Coder died on API 529 Overloaded → classified INFRASTRUCTURE → resumed
- **Analysis** — infra vs real bug: INFRASTRUCTURE (server-side 529, transient; nothing code-related). Tree verified clean at `becaef7` (no partial edits to reconcile). Remedy: resumed the same Coder agent with its context intact and the full six-fix briefing restated; no code fix, no relaunch-from-scratch needed.
- **Result** — `in_progress` (Coder resumed).

## 2026-09-03T13:52:40-04:00 — R3 fixes COMMITTED (`3d14856`) on Yixun's approval → round R3 CLOSED → pushed
- **Version Control** — ORBITRIR main bffe709 → ed114ac (C7a) → 23cb7f4 (C7b) → becaef7 (C7c) → `3d14856` (R3-fix), pushed to origin/main. Suite 188/188; window whitespace clean.
- **Analysis** — evaluation side complete: helpers, wiring, trained-as guard, hardened authority/EMA/identity. Planner additions this round: real-checkpoint probe (released + BF40) as ladder rung 3; 2-line fork-history reword in test docstrings. Remaining: C8-C12 (R4) + smoke + pinned two-cell acceptance.
- **Next** — C8 launched per Yixun's standing instruction (approve R3 fixes → push → C8).

## 2026-09-03T13:53:10-04:00 — Coder round C8 launched (first of the R4 window)
- **Goal** — C8: --max-steps via build_trainer_kwargs/construct_trainer boundary (verbatim from the e85ebde snapshot; defaults.ini max_steps=1000000), test_train_max_steps.py trimmed per D8 (drop revert-guard + import-side-effects; keep compact 14-key literal parity per Codex R1 ruling).
- **Acceptance criteria** — suite ≈196 green; parity diffs vs e85ebde EMPTY (or enumerated); re-hardcode mutation fails override tests; git diff --check clean; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C8 diff to Yixun → approval → commit → C9 → C10a/b → C11 → C12 → R4 review → push → smoke + two-cell acceptance.

## 2026-09-03T15:14:42-04:00 — C8 COMMITTED (`619084a`) with the README fold-in
- **Result** — `passed`. Parity vs e85ebde byte-exact for train.py/defaults.ini; README HAA recipe now passes --max-steps 1000 (stale edit-the-code instruction removed). Suite 196/196.
- **Next** — Coder round C9 (--sync-batchnorm).

## 2026-09-03T15:15:15-04:00 — Coder round C9 launched
- **Goal** — C9: --sync-batchnorm (verbatim e85ebde→f362673 delta; result byte-identical to f362673 blobs modulo one reworded attribution parenthetical), test_train_sync_batchnorm.py trimmed per D8 (drop prefigure canary + revert guard + import-side-effects; keep CLI override per Codex R1 ruling + all behavioural tests incl. only-key-added and smuggle guard).
- **Acceptance criteria** — suite ≈210 green; parity vs f362673 empty or reword-hunk-only; both mutations (guard no-op; always-add-key) red→restored; git diff --check clean; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — ladder → C9 diff to Yixun → approval → commit → C10a/b (grad ckpt) → C11 (FA config) → C12 (README) → R4 → push → smoke + acceptance.

## 2026-09-03T15:29:11-04:00 — C9 COMMITTED (`10d9895`) with the letter-tag strip
- **Result** — `passed`. _as_bool("") silent-False kept as-run per Yixun. Next: C10a.

## 2026-09-03T15:29:43-04:00 — C9 AMENDED to `e9a4139` (tag strip had silently failed)
- **Analysis** — my strip script died on a malformed regex; the shell chain still committed 10d9895 WITH the tags, violating the approved condition. Caught immediately (the residual grep printed 5). Fixed with exact-string replacements (5 docstring tags), pytest re-run green, commit AMENDED (unpushed, so history stays one-commit-as-approved); commits row corrected to `e9a4139`. Lesson logged: never chain a commit after an edit script without asserting the edit's own success first.
- **Result** — `passed` (suite-file 27/27).

## 2026-09-03T15:30:22-04:00 — Coder round C10a launched
- **Goal** — C10a: grad-ckpt surface from f59f5a4 conditioners.py (byte-exact code; review-attribution paragraph reworded impersonally; NO token_pool/cyl_vit) + stub-backbone test file part 1 (counting shim, tiny ViT, atol=0 equivalence, fail-closed family, idempotency, state_dict identity). DINOv3-backed tests = C10b.
- **Acceptance criteria** — suite ≈235 green; parity hunks = reword + later-work absences + C1 revision hunk only; 3 mutations red→restored; no worklog/BV.json references; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).

## 2026-09-03T16:22:01-04:00 — C10a COMMITTED (`157c5f4`); Coder round C10b launched (tests-only, real DINOv3)
- **Goal** — C10b: 7 DINOv3-backed integration tests built from the IN-TREE FLAC_AR.json conditioning block + in-memory flag injection (Codex R1 finding 2a); cache-guarded but MUST PASS (not skip) on this box; gradient identity allclose(1e-6/1e-5) with >100 tensors; layers[1:]-wrap mutation check.
- **Acceptance criteria** — 7/7 PASSED zero SKIPPED; full suite ≈239 green; conditioners.py untouched; no banned refs; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running; DINOv3 CPU forwards+backward make this the slowest round, expect 20–35 min).

## 2026-09-03T16:39:14-04:00 — C10b COMMITTED (`17ecd46`); Coder round C11 launched
- **Goal** — C11: FLAC_AR_FA.json (= FLAC_AR.json + exactly 4 leaf deltas) + D10 revision pin 114c1379… in BOTH configs' ViT blocks + pure-JSON type-strict test file (flatten keyed by conditioner id; guard-compat via resolve_conditioning_options; revision pin test). CRITICAL integration check: the C10b DINO tests build from FLAC_AR.json — the pin must resolve from the local cache (7 passed, 0 skipped) or the Coder stops and reports.
- **Acceptance criteria** — sorted-JSON diff shows exactly the 4 additions; suite ≈243 green incl. DINO file 7/7 post-pin; 3 mutations red→restored; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).

## 2026-09-03T21:08:50-04:00 — C11 COMMITTED (`8fc6d72`, literal hash pin mutation-checked inline); Coder round C12 launched (final content commit)
- **Goal** — C12: README "Frame-Averaged (Yaw-Equivariant) Conditioning" section (method, 2-GPU recipe with SyncBN/grad-ckpt rationale, eval with inherit-guard + bf16 protocol note, rotation check, supported-API subsection) + 4-line raw-conditioner comment in generation.py. Every command verified against real argparse.
- **Acceptance criteria** — suite unchanged green; commands parse (bash -n + --help cross-check); no banned refs incl. "B-F"; no commit by the Coder.
- **Result** — `in_progress` (Opus 5 Coder running).
- **Next** — C12 diff → approval → commit → R4 review (C8..C12) → fixes → push → smoke + two-cell acceptance.

## 2026-09-03T22:09:30-04:00 — C12 COMMITTED (`da370c2`, with the variants-table row) — ALL 12 CONTENT COMMITS DONE; R4 Codex review LAUNCHED
- **Version Control** — ORBITRIR main 3d14856 → 619084a (C8) → e9a4139 (C9) → 157c5f4 (C10a) → 17ecd46 (C10b) → 8fc6d72 (C11) → **da370c2 (C12)**. Unpushed; push when R4 closes.
- **Command / Validation** — codex exec gpt-5.6-sol xhigh, backgrounded, verdict → orbitrir_fa_port_codex_code_r4_review.md. Briefing: final round + completeness-critic hat; focus (a)-(e) incl. README-vs-code accuracy, unpinned variant configs adjudication, and GO/NO-GO for the execution phase (smoke + pinned two-cell acceptance + guard negative control).
- **Result** — `launched`.
- **Next** — on verdict: triage → fix round if needed (diff to Yixun per gate (b)) → push → execution phase.

## 2026-09-03T22:13:22-04:00 — C13 (`9ce4c4a`): OrbitRIR identity edits per Yixun (title/byline/badges/fa_cond.png)
- **Note** — docs+asset commit outside the running R4 window; will be disclosed to the reviewer at R4 closure (or riding the R4-fix round if one happens). fa_cond.png (1.1MB) was placed in assets/ by Yixun himself.

## 2026-09-03T22:17:16-04:00 — C14 (`16d4459`): anonymous-review README per Yixun
- **Note** — README-only per instruction. Remaining identity/HAA surfaces OUTSIDE the README flagged to Yixun: download_weights.sh contains the author's HF handles; the repo still tracks HAA dataset configs/splits (src/configs/dataset_configs/HAA/, data/HAA/, FLAC_HAA_finetune.json) and AGREE HAA branches; badges/citation now gone from README only.

## 2026-09-04T09:40:18-04:00 — R4-fix (`73d30eb`) + C15 (`834dfa1`) committed; R4-closure re-verify next
- **Result** — `passed`. R4's five findings fixed and verified (env-leak spot check via two-module pytest session; typed-diff; README arithmetic; whitespace range-clean). Anonymization: items 2+3 done + Yixun's mid-turn README deletions (Acknowledgements / AGREE T&F / Baselines); item 1 awaits an anonymous weights URL from Yixun (download_weights.sh still carries the author's HF handles until then).
- **Next** — focused Codex re-verify closing R4 (covers 73d30eb + C13 + C14 + C15) → push origin/main → symlinks → smoke → pinned two-cell acceptance + guard negative control.

## 2026-09-04T09:52:52-04:00 — R4 CLOSED (`3407814`) → ENTIRE PORT PUSHED to origin/main
- **Version Control** — origin/main now = `3407814`; 23 commits on top of upstream ead8bbd. Code phase COMPLETE: C1–C12 + C13/C14/C15 identity+anonymization + 4 review-round fix commits; 4 Codex rounds closed (R1–R4 + focused closure re-verify); suite 243/243.
- **Analysis** — closure-fix loop closed on Planner verification (SOP-sanctioned for re-verify): tests green, whole-range whitespace clean, amandine/HAA residual sweeps clean (remaining HAA strings are generic dataset support in upstream metrics code, reviewer-classified as fine). Open item: download_weights.sh anonymous URL (Yixun).
- **Next** — execution phase per plan §3.4-3.6: symlinks → storage-light smoke (3-step DDP+SyncBN) → pinned two-cell acceptance + rot90 C4-invariance cell + guard negative control (refusal + override off-diagonal reproduction).

## 2026-09-04T09:57:19-04:00 — author-info sweep DONE (Yixun request); execution runner launching
- **Sweep results (tracked content @1f6e3bf):** paths CLEAN; contents CLEAN except the known download_weights.sh open item; AGREE subtree checked separately for emails/names. **FLAG: git commit metadata carries both author identities (upstream AmandineBtto + Yixun-Hu) — any share that includes .git (incl. the GitHub URL) reveals them; anonymous submission must go through an anonymized mirror or a zip without .git, or a history rewrite (all SHAs change). Yixun's call.**
- **Acceptance criteria for the runner (pre-registered):** [1] smoke rc=0, reaches 3 optimizer steps, no OOM/NaN, SyncBN active on 2 GPUs; [2] negative control rc!=0 with the trained-as refusal message, no metrics JSON written; [3] K=8 s42 within tol of 8.1902/0.9804/38.8113/0.3333/R5.3022; [4] K=1 s42 within tol of 9.4859/1.0547/41.2371/0.3284/5.2391 (tol T60 .005 / C50 .001 / EDT .02 / FD .001 / R@1 .02); [5] rot90 K=8 ≈ rot0 (C4 invariance, deltas reported); [6] off-diagonal override runs, is disclosed in its record, and lands near the registered 2x2 row (~10.65/2.08/80.9/R0.7, informational). Full split (6,337/17) everywhere; bf16 conditioning autocast; per-scene means.
- **Result** — `launched` (background runner, timestamped log in this folder).

## 2026-09-04T10:53:55-04:00 — EXECUTION PHASE COMPLETE — ALL PASS — exp_22 CLOSING
- **Result** — `passed`. Smoke 3/3 steps (losses 2.36/2.61/2.44); guard refusal correct (no artifact); K=8+K=1 acceptance |Δ|=0.0000 on all 10 pins; rot90 C4-invariance ≤0.0033; off-diagonal override reproduces the historical row exactly with disclosure. Runner cosmetic noted: rc= lines report tail's rc (pipeline); outcomes proven by artifacts + in-log evidence instead. GPUs released.
- **Next** — results.md + analysis.md written; closure report to Yixun. Open: weights URL, .git metadata decision, optional 5-seed matrix.
