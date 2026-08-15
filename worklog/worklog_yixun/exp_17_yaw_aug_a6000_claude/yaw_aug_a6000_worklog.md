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
