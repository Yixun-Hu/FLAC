# exp_17 yaw_aug_a6000 — lab notebook (append-only)

## 2026-08-14T23:34:00-04:00 — scaffold and provenance audit

- **Goal** — Define the paper-facing 2×A6000 random-yaw augmentation arm after loading the SOP, all standing announcements, exp_07 P1/B-F provenance, and exp_15's reviewed augmentation design.
- **Hypothesis** — Reusing the reviewed exp_15 augmentation implementation with a P1-derived single-delta config will isolate training-time yaw augmentation under the same hardware/rung/budget as the existing B-F/P1 comparison.
- **Version Control** — Planning performed from branch `check-equivariance-necessity`; no implementation or launch yet. Proposed clean training base is exp_15 round-1 closure `58d0b887`, before later unrelated training-path changes.
- **Command / Validation** — Confirmed `FLAC_AR_YAWAUG.json` and `FLAC_AR_BVp1.json` are semantically identical except `training.yaw_aug`; P1@40k checkpoint sha256 is `c4c678826cddda37fa4977926aadee530afd037b3abb110918b52a342ce9845c`. The live checkpoint-curve chains are healthy and remain first in GPU order.
- **Result** — `in_progress`; plan drafted, no GPU allocated and no training code/config created.
- **Analysis** — Existing exp_15 is 8×L40/micro-8 against exp_11 VANL and is not a matched control for the 2×A6000 legacy FA result. It remains scientifically useful as a recipe-transfer observation, not the primary row here.
- **Next** — Opposite-family plan review; revise findings; surface the final design before implementation/launch. Training may start only after the current curve chain releases both GPUs.

## 2026-08-15T12:51:17-04:00 — plan review adopted and local A6000 execution authorized

- **Goal** — Close plan review and resolve its sole blocker before implementation.
- **Change** — Yixun supplied the archived Opus 5 review and directed that Neuronic be ignored in favor of the local 2×A6000 run. Plan amended to replace nonexistent `58d0b887` with audited base `41aa31dc0f5787019f26912654e4c5a14be7feeb`, document default-path/RNG parity, register the regularization interpretation, add a >55 h smoke stop threshold, declare eval cap 64, and bind both train/eval SHAs.
- **Version Control** — Shared checkout `check-equivariance-necessity`; reviewed base `41aa31dc0f5787019f26912654e4c5a14be7feeb`. Implementation will occur in an isolated `exp-17-yawaug-a6000` worktree and final training will bind its reviewed commit.
- **Command / Validation** — Confirmed both RTX A6000 cards idle at authorization (GPU0 11 MiB, GPU1 27 MiB; no compute processes). Confirmed `58d0b887` is absent. Verified the supplied review artifact is tracked at base HEAD.
- **Acceptance criteria** — No implementation begins until blocker B1 is explicitly resolved in the plan; no full run launches before TDD, independent code review, static/real-data checks, and two-GPU smoke pass.
- **Result** — `passed`; plan review closed and implementation authorized.
- **Analysis** — Current-HEAD isolation is safer than an unverifiable historical pin and prevents the shared checkout from mutating beneath a 40k run.
- **Next** — Create isolated worktree; implement config/tests/launcher; run review and validation ladder; launch if green.

## 2026-08-15T12:55:21-04:00 — plan revised to Rev 2 against the Opus-5 verification review
- B1 RESOLVED: `58d0b887` does not exist on this remote; base is now current HEAD in an isolated worktree, with the default-preservation argument for exp_14/exp_16 written out explicitly. **Supersedable if Yixun supplies the real SHA.**
- R1 regularization-confound reading registered BEFORE any metric is seen (0° gain vs rotated gains). R2 eval flags corrected (`--frame-avg-angles` dropped as inert for vanilla; `--frame-avg-max-fwd-samples 64` declared per announcement 06). R3 smoke abort threshold pre-registered (>55 h projection ⇒ stop, do not launch). R4 GPU policy aligned to co-tenancy + free-VRAM floor. R5 manifests bind training AND evaluation source SHAs.
- Sections verified correct in review (schema, draw key, pose fields, img_w, physics, config base, flags, existing grids, curve workers, P1 resume caveat) left unchanged.

## 2026-08-15T13:20:00-04:00 — implemented from scratch on `exp17-yawaug-scratch`, visibly, at Yixun's request

- **Goal** — Yixun asked to watch the code being written ("我的核心诉求是想要看到你是怎么编辑代码修改的"), so the round was rebuilt step by step in the MAIN checkout on a fresh branch `exp17-yawaug-scratch` rather than in the pre-existing `exp-17-yawaug-a6000` worktree, whose branch could not be checked out in his IDE while a worktree held it.
- **Step 1 — arm config.** `FLAC_AR_YAWAUG_A6000.json` built by BYTE insertion into a copy of `FLAC_AR_BVp1.json`, not by re-serialising a parsed object, so `diff` itself proves the delta count.
- **Step 2 — 18 contract tests** (`src/tests/test_yaw_aug_a6000_arm_config.py`), RED→GREEN. Read as bytes, never text: universal-newline decoding would hide a CRLF drift.
- **Step 3 — launcher + 25-case guard suite.** Every gate driven by the real launcher, asserting exit code AND reason text.
- **Step 4 — Codex r1 review** → REQUEST-CHANGES, 4 launcher blockers + 2 test + 4 guardtest findings (`yaw_aug_a6000_codex_code_r1_review.md`).

## 2026-08-15T13:40:00-04:00 — grad-ckpt OFF (registered deltas 2/3) + Codex r1 applied

- **Change** — Yixun freed all A6000 VRAM for this arm and directed "close the ViT gradient check". Both `gradient_checkpointing` flags flipped `true → false`, so the arm is now the control plus THREE registered deltas, not one.
- **Why it does not break the control** — exp_07 measured ViT grad-ckpt ON vs OFF as producing bitwise-identical parameter gradients (210 tensors, max abs diff 0.0; `state_dict` sha256 unchanged), pinned by `src/tests/test_vit_gradient_checkpointing.py`. It is a VRAM-for-time trade, not a numerics change.
- **Tests 18 → 21** — pinned the P1 control sha256 (`733ca52b…a49d8`) so *coordinated* drift of control and arm can no longer preserve the "control plus deltas" claim; forward-construct the arm bytes from all three deltas; assert grad-ckpt OFF explicitly; replaced the vacuous width guard (it had only asserted 256 ≠ 512, and would have passed with the pin deleted) with one that calls the real checker from both sides, plus a non-vacuity case. **21/21.** The placeholder SHA was left in deliberately for one run first, to confirm the new pin actually fails.
- **Launcher (all four r1 blockers)** — (1) sha256 content pins on the 7 files that define the treatment + clean-tree gate under `src/`; (2) banner matched as a WHOLE LINE against `diffusion.py:407` — the old substring grep was satisfied by the launcher's own preflight echo, so it could pass with the treatment dead — and the preflight is now worded so it cannot collide; (3) `MAX_PROJECTED_HOURS`/endpoint/cadence made non-overridable and FULL now REQUIRES a smoke log carrying banner + `SMOKE VERDICT: PASS`; (4) SMOKE cadence 1,000,000 ≫ its 25-step endpoint, with zero checkpoints asserted after the run.
- **Guardtests 25 → 41** — accept cases now stop at a real `DRY_RUN=1` boundary and assert the ACTUAL `train.py` ARGV (r1: they had asserted a preflight paraphrase, so a wrong `--max-steps` stayed green); new G/H/I sections cover source pins, FULL-requires-smoke, banner self-satisfaction, R3 wiring, smoke-checkpoint fatality; restoration failure is now fatal. **41/41.**
- **Cleanup bug found by the suite itself** — two launcher runs inside the same second share a timestamp and therefore a log path, so per-case counting never moved and 5 dry-run logs leaked into the directory the smoke-evidence gate globs. They could not have spoofed the gate (no banner, no PASS), but cleanup is now a set difference.
- **Commits** — `c5e44a7`, `b2af05b`, `9ba9de7` (pushed).

## 2026-08-15T13:53:51-04:00 — SMOKE PASS

- Banner FOUND as an exact whole line; **0 checkpoints**; no nan/inf; `max_steps=25 reached`, rc=0.
- R3 upper bound **38.2 h ≤ 55 h → PASS**. Steady state from tqdm stamps over steps 10→25 is **2.200 s/opt-step → 24.4 h** for 40,000 steps (P1 vanilla was 3.86 s/step = 42.9 h; the 1.75× is the grad-ckpt change, not the augmentation).
- Topology confirmed: 2 NCCL ranks, 50.3 M trainable / 30.9 M frozen, DINOv3 ViT-S/16 21.60 M trainable, 291,210 files / 243 subfolders, 4,550 steps/epoch → 40k ≈ 8.8 epochs.
- **FULL held pending Codex r2**, deliberately: once FULL starts, the launcher is train.py's parent for ~24 h and must not be edited, so any r2 launcher finding has to land first.
