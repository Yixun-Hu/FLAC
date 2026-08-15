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
- **Why it does not break the control** — grad-ckpt recomputes activations in the backward pass, so it is a VRAM-for-time trade, mathematically the same gradients. ⚠️ **Corrected after Codex r2:** I first wrote that this is "bitwise identical (210 tensors, max abs diff 0.0), pinned by `test_vit_gradient_checkpointing.py`". The test actually asserts `torch.allclose(atol=1e-6, rtol=1e-5)` over ≥100 tensors on an **fp32 CPU** probe and only *prints* max_diff; the 210/0.0 figure is an exp_07 worklog observation, not a CI-pinned invariant, and this arm runs bf16-mixed CUDA. The delta remains admissible, but the evidence is allclose-grade, not bitwise-grade.
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

## 2026-08-15T16:10:00-04:00 — our FULL stood down; the concurrent arm is the experiment

- **Discovery** — while our FULL launch sat blocked by a permission gate, the `exp-17-yawaug-a6000` worktree launched its own exp_17 FULL at **14:54:52** (train.py 14:55:14), grad-ckpt **ON**, otherwise the same recipe (40,000 / 32×2×1 / SyncBN / bf16 / seed 42 / cadence 2,500). Not launched by us; our attempt never created a process.
- **Rate claim RETRACTED** — I had reported "grad-ckpt off buys 3.86 → 2.200 s/step, saving 19 h". That was wrong. Compared in the SAME step window (10→25, both still warming up): ON 1.933 s/step vs our OFF 2.933 s/step; the ON arm's true steady state over steps 150→266 is **1.966 s/step → 21.8 h**. Our number came from a 25-step smoke whose only available window is entirely inside warm-up (dataloader spin-up, cold caches, cudnn autotune), and one of those smokes was additionally contending with two `eval_FLAC.py` processes at 22–23 % utilisation. **A 25-step smoke cannot measure steady-state throughput.** The other worktree's launcher had already solved this correctly with a windowed tqdm rate (`tqdm_window_5_25`) that forbids a wall-clock fallback.
- **Decision (Yixun)** — wait for the running arm. It is also the cleaner design: strictly single-delta against P1.
- **Codex r3 independently supports it** — r3 rejected the surviving claim that checkpointing "cannot move the trajectory": a one-step fp32 CPU allclose probe does not establish 40k-step bf16 CUDA trajectory equivalence, so checkpointing OFF is a **disclosed numerical confound**. The running arm keeps it ON and matches P1 exactly, so it carries none of it.
- **Read-only audit of the running arm's launcher** — it binds MORE pre-run pins than ours (commit, `defaults.ini`, AR split, VAE, ViT snapshot, seed-42 initial `state_dict` hash) and has a FULL **banner watchdog** that fails if a training step precedes the exact banner — earlier and stronger than our post-hoc grep. Its one gap is post-run: the FULL path ends at `exit "$RC"`, and `postrun.py` defines only `smoke_problems()`.
- **Gap closed from outside** — `src/tools/exp17_full_audit.py` (+ 16 TDD tests): a pure function over (log text, checkpoint names, rc), so it is testable without a 21 h run and appliable read-only to a run we did not launch. Checks the endpoint marker *with Lightning's backticks*, `*step=40000.ckpt`, all 16 cadence checkpoints, the banner as a whole line, 2-rank topology, and finite loss. Verified against the live log: banner and topology matched real data; the three problems reported are exactly "not finished yet". This also covers r3's finding that NaN/Inf checking was SMOKE-only — a FULL that diverges late would otherwise reach 40k and pass every gate.
- **Our launcher/guardtests are RETIRED** pending the r3 blockers (HEAD binding, smoke-evidence revision binding, `pipefail`+`tr|grep -q` SIGPIPE, FULL NaN check, guard-suite H/I deletion-vacuity, concurrency-unsafe cleanup). They must not be reused as-is.
- **Process slip** — Codex r3 stalled 46 min at 0 % CPU on `Reading additional input from stdin...` because I omitted `< /dev/null`, which CLAUDE.md documents. Killed and relaunched correctly.
