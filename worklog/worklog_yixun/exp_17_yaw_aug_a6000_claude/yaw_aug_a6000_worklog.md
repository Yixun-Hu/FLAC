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

## 2026-08-16T12:39:13-04:00 — FULL COMPLETE, audit PASSED

- `exit rc=0`; 21 h 44 min wall; **16/16** cadence checkpoints (2,500 → 40,000), 11 GB; final train/loss 0.409 from 2.47; zero Traceback/OOM/Killed/RuntimeError across the whole log.
- Completion audit: **COMPLETE** — endpoint reached, 16 cadence checkpoints, treatment banner present as a whole line, 2 ranks, all logged losses finite, log bound to the audited directory, foreign `exit rc=0` parsed rather than assumed.
- **The r1 fix was load-bearing.** The real marker is `…train/mse_loss=0.409]\`Trainer.fit\` stopped: \`max_steps=40000\` reached.` — appended to the tqdm line, not standing alone. The whole-line match would have rejected this valid run; my fixture had put the marker on its own line, which is why the tests were green while the tool was broken.
- Measured rate 1.931–1.966 s/step throughout, no thermal or contention decay.

## 2026-08-16T12:47:17-04:00 — C4 rotation grid launched (127 of 128 cells)

- **Probe first.** One cell (S40000/K=1/0°) run end-to-end before committing the queue: rc=0, **389 s** — so the real cost is 6.5 min/cell, not the 11 min estimated from historical logs, and the grid is ~6.9 h rather than ~12 h. The probe's artifact passed our own admission gate (`cond_method=vanilla`, `cond_autocast=bf16`, `rotate_deg=0.0`, 11 metrics), so the queue skipped it: 128 → 127.
- Probe numbers (S40000, K=1, 0°): T60 9.366 / C50 1.083 / EDT 41.902 / FD 0.322 / R@1 5.034 / R@5 15.922 / R@10 22.992.
- Both gates fired correctly in sequence: gate 1 (training pid dead), gate 2 (audit COMPLETE with the parsed foreign rc=0). Symlink farm built, 16 links, provenance printed — every write lands in our namespace, the foreign tree is read-only.
- ⚠️ Cosmetic: the runner's queue banner still says "~11 min each => ~11 h" from the pre-probe constant. Not corrected mid-run — bash reads a script as it executes, so editing a running launcher corrupts it. Display only; execution unaffected.
- `src/tools/exp17_rotation_table.py` (+11 TDD tests) written while the grid runs: one row per COMPLETE (step, K) orbit, reporting the value at 0° and the **C4 spread** (max − min). An incomplete orbit yields NO row by design — a 3-of-4 max-min is a smaller, flattering number, not the C4 spread. Cells whose recorded protocol disagrees, or whose recorded angle contradicts their filename, are hard errors: a silently permuted orbit would produce a plausible spread that means nothing.

## 2026-08-16T17:00:00-04:00 — P1-control kit review-clean (r1 REQUEST-CHANGES → r2 all FIXED, no new blockers)

- Seat note: main session model changed Opus 5 → **Fable 5** at 14:43; four handoff docs refreshed (`faf312e`). Everything below this entry is Fable 5's authorship.
- **Grid mid-run findings fixed live** (no GPU time lost — aggregation-layer only): eval_FLAC appends its own rotation suffix (`..._rot90_seed42_rot90.json`) so the end-anchored name parser saw only the 0° quarter; metric JSONs carry bare keys (`T60`) not stdout labels. Fixtures now mirror real artifacts byte-for-byte (`b5691e8`). Same failure class as the termination-marker bug: **idealized fixtures pass while the tool is broken.**
- **P1-control C4 grid launch-ready** pending Yixun's GO: planner parameterized by arm (default byte-identical — regression-pinned; the running grid's final pending-check is unaffected), control runner with the SHARED `.roteval.lock` (r1: private lock + pgrep snapshots was one-sided and racy), exact single-match checkpoint resolution (P1's 2500–40000 window verified 16/16 unique `.ckpt`; the "duplicates" were historical eval JSONs in the same archive dir), P1 config byte-pinned. Live-tested: refused at the lock while the YAWAUG grid runs.
- **Estimand clarified and labeled** (r1 blocker 3): grid cells carry GLOBAL (sample-weighted) means — verified identical to the flat key every `model_comparison.md` raw JSON uses (probe JSON T60 == stdout T60), so cross-row comparability holds; but it is NOT the per-scene convention my test docstring had claimed. Table output now states this on every render. `load_cells` is arm+seed-scoped with duplicate-identity hard errors (r1: last-write-wins could fabricate a complete orbit from two half-arms).
- Commits `0f24a41` `c39ad12`; reviews archived (`…p1ctrl_r1_review.md`, `…p1ctrl_r2_review.md`). 47 planner+table tests green.
- Mid-grid sneak peek (17 orbits): ΔT60 0.02–0.18 throughout, ΔEDT 0.4–1.7 — an order of magnitude below the vanilla anchor's 90° degradation (exp_07 A6: EDT +5.83); T60@0° at S20000 = 8.533 already inside P1@40k's band. Single-seed, no same-protocol control yet.

## 2026-08-16T23:58:00-04:00 — (a) ACCEPTED by Yixun; (b) P1-control grid LAUNCHED; (c) acceptance pending

- **Yixun:** "confirm a is done, continue on b"; separately: "Later on I will check results of c and confirm c is done" — (c) 45° probe results delivered but acceptance OPEN.
- (a)+(seen) delivered with full ±std on every column (chat rendering had truncated them — data was always complete in `results_yaw_aug_a6000_endpoint_rows.md`).
- Branch rebased onto latest `check-equivariance-necessity` (evidence for other sessions' rows restored; HANDOFF/tracker conflicts merged keeping both sessions' states per precedent; force-with-lease push). Found and logged: **table regeneration is cluster-only** — exp_11 rows' validator pins cluster-absolute ckpt paths; Yaw-Aug rows staged in the generator for the next cluster-side regen.
- **(b) launched 23:55:48** after two false starts: (i) first attempt refused because my own YAWAUG progress-monitor's command line contained the runner's name and `pgrep -f` matched it — the monitor also never detected the grid's end for the same reason; monitor stopped, relaunched with a `[n]`-bracket pattern that cannot self-match. (ii) GPU check initially flagged "processes on both cards" — they are user kevinwang's `ga_flow_toy` jobs (~3–5 GB, 99% util), NOT ours; co-tenancy per standing policy with disclosure. **Expect the 6 h nominal queue to stretch (rough guess 8–12 h) under contention.**
- All 16 P1 checkpoints farmed with exact-single-match resolution; 128 control cells queued (`exp17_P1CTRL_*`, arm-scoped).
