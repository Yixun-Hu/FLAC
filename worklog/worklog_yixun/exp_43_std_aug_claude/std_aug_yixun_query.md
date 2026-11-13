# exp_43 std_aug — Yixun's driving queries

## Query 1 — 2026-11-12 ~10:52 EST (main session, sent while exp_42's plan r1 was being presented)

**Verbatim:**

> 我们的yaw-augmentaion 要用类似panoformer这篇文章的方法，重新跑一遍：https://github.com/zhijieshen-bjtu/PanoFormer/blob/main/PanoFormer/dataload.py#L82 就是这个里面好像用了C4 and mirror flip，你需要做这个实验

**Summary:** re-run the Yaw-Aug baseline with an augmentation modelled on PanoFormer's data loader (the linked line), which Yixun recalls as "C4 and mirror flip".

**What the linked code does (read 2026-11-12; `PanoFormer/dataload.py` on `main`, sha256 `fefe8243…518d0`, lines 82–94):**

```python
if self.is_training and self.yaw_rotation_augmentation:
    # random yaw rotation
    roll_idx = random.randint(0, self.w//4)
else:
    roll_idx = 0
rgb = np.roll(rgb, roll_idx, 1)
gt_depth = np.roll(gt_depth, roll_idx, 1)

if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
    rgb = cv2.flip(rgb, 1)
    gt_depth = cv2.flip(gt_depth, 1)
```

i.e. (1) a horizontal roll by a uniform integer number of columns in [0, W/4] inclusive — a random yaw between 0° and 90° in single-column steps (129 offsets at W = 512), **not** the four quarter turns of C4; (2) a left–right mirror flip with probability 0.5, applied after the roll; (3) colour jitter (RGB only; not applicable to FLAC's depth panoramas).

**Assumption / hypothesis (surfaced by the Planner, not Yixun's words):** the purpose is a "standard augmentation" baseline for the OrbitRIR rebuttal that can be cited to an existing panorama paper. Because the code differs from Yixun's recollection, which augmentation is trained is an explicit decision for Yixun (literal PanoFormer; C4 × mirror; full-circle roll × mirror) — not chosen by the Planner.

**Why this experiment needs to run:** no FLAC checkpoint was trained with a mirror flip; the existing Yaw-Aug arms are a full-circle roll (exp_15 YAW) and its 16-heading restriction (exp_40 C16). The mirror needs new code: the depth panorama stores per-pixel 3D vectors (`depth` is `[3, H, W]`), so a flip is a column reversal plus a reflection of those vectors and of every pose vector (`src/data/yaw_rotation.py::rotate_scene_metadata` is the rotation counterpart).

## Query 2 — 2026-11-12 ~10:53 EST (sent a minute later, same turn)

**Verbatim:**

> 然后orbitrir还需要有这个实验：“Training efficiency experiment (but just incremental increasement, we should have a standard augmentation methods)” 我们的方法应该能够在loss （train and val）上有更好的convergence

**Summary:** a training-efficiency experiment for OrbitRIR: compare the convergence of the training and validation loss of OrbitRIR against a standard augmentation method; Yixun expects OrbitRIR to converge better.

**Assumption / hypothesis (Planner):** the "standard augmentation method" of this experiment is the arm of Query 1. The expectation "OrbitRIR converges better" is a hypothesis to be tested, not a result to be produced; the comparison protocol (which loss, on which data, at which headings) is fixed before any curve is looked at. Whether this is part of exp_43 or its own experiment (exp_44) is a decision for Yixun.

**Facts relevant to it (2026-11-12):** every 40k recipe run so far was launched with `--val-every -1 --val-dataset-config ""`, so no existing run logged a validation loss; the wrapper can log `val/loss_<t>` and `val/avg_loss` when validation is enabled (`src/training/diffusion.py:1054-1076`), and `train/loss` is logged every step (`:967`). OrbitRIR's training logs and intermediate checkpoints are on neuronic, not on this machine.

**Why this experiment needs to run:** no loss-convergence comparison between OrbitRIR and an augmentation baseline exists; the existing evidence compares endpoint metrics (exp_38, exp_40) only.

## Query 3 — 2026-11-12 ~11:10 EST (answer to "Which augmentation should the new standard-augmentation baseline train with?")

**Verbatim:**

> Full roll + flip

**Summary:** decision E1 = option C: the baseline is trained with a uniformly random full-circle yaw roll followed by a left-right mirror flip with probability 0.5 — the UniFuse loader's augmentation (`UniFuse/datasets/matterport3d.py`: `roll_idx = random.randint(0, self.w)`, then `cv2.flip(…, 1)` with p = 0.5), of which PanoFormer's loader is the variant with the roll range cut to W/4.

**Assumption logged (Planner):** "full roll" = FLAC's existing continuous Yaw-Aug offset stream (uniform over the 512 column offsets), unchanged; the only new operation is the mirror flip. Everything else (recipe, data, evaluation protocol) follows exp_40. Still open and not assumed: how validation loss is obtained for the convergence comparison (E2), which experiment gets the two A6000s first (E3), and the rebuttal deadline.

**Why:** Yixun wants the augmentation baseline to be a recognised standard panorama augmentation rather than a project-specific one.

## Query 4 — 2026-11-12 ~11:20 EST (sent while the plan was being written)

**Verbatim:**

> 然后这个yaw-augmentaion的results大概需要多久能够完成training and eval？

**Summary:** how long until the new (roll + flip) Yaw-Augmentation arm has finished training and evaluation.

**Assumption (Planner):** "results" = the θ = 0 rebuttal row (K ∈ {1, 8}, evaluation seeds 42–46) of the 100 % / 40k run; the yaw-robustness cells are reported as a separate, later stage. The estimate assumes both A6000s are free of co-tenants and that exp_43 gets the cards before exp_42 (decision E3, not yet taken).

**Answer given (basis: exp_40's measured durations, plan §12):** training 21.6 h; end audit + admission ≈ 0.5–1 h; E0 + 10 T cells + verify / collect / table ≈ 2.8 h; engineering before launch (Codex plan rounds, implementation, three code rounds, CPU suite, smokes) ≈ 6–9 h → rebuttal row ≈ 32–34 h after plan approval; the 56 robustness cells ≈ +7 h.

**Why:** Yixun is scheduling the rebuttal around this number.

**Correction (2026-11-12 11:48 EST, after Codex plan round 1):** the 6–9 h preparation figure was optimistic (Codex: 12–18 h; plan r1 with scope cuts: 10–14 h). Revised estimate: rebuttal row ≈ 35–40 h after plan approval; the 56 robustness cells ≈ +7 h. Reported to Yixun with plan r1.

## Query 5 — 2026-11-13 ~00:50 EST (approval of plan r1; priorities across experiments)

**Verbatim:**

> exp_42: 按照你的recommendation批准
> exp_43: 按照你的recomendataion批准，尽快开始按照panoformer的方法重新跑一遍，这个实验是最高优先级，先安排这个实验
> exp_44: 开始做，就先按照A6000和neuronic上的L40都include进来的方式考虑
> exp_36: C4 augmentation的model其实就是exp_43的没有flip的版本，可以先不做，先等exp_43
> exp_37: 这个应该不用跑了

**Summary:**
- exp_43: plan r1 (`plan_std_aug.md`, sha256 `f9182f47…7769` at approval) is **APPROVED** with every recommendation of §11: E2 post-hoc fixed loss protocol (exp_44), E3 exp_43 takes the cards before exp_42, E4 Stage A then Stage B, E5 label `Roll + Flip (UniFuse-style)` / run id `exp43_RF_a6000_2x32`, E6 unattended chain yes, E8 no mirrored-evaluation cells, E9 no same-pin YAW control. E7 (rebuttal deadline) was not given; E10 (external AR frame documentation) none supplied, so the frame audit is the only evidence. Implementation starts immediately; exp_43 is the highest priority and is scheduled first.
- exp_42: plan r2 approved with recommendations D1–D8 (logged in the exp_42 query log).
- exp_44: start the work (scaffold, survey, plan); design the comparison so that runs from both this machine (A6000) and neuronic (L40) are included. Its plan still needs approval before implementation.
- exp_36: deferred until exp_43 has results. exp_37: not to be run. Both belong to another session (`orbitrir-review`); this session only records the decisions and does not touch that session's processes.

**Assumptions logged (Planner):**
1. "按照panoformer的方法" = the approved plan's treatment: full-circle roll + mirror flip with p = 1/2 (decision E1 of Query 3, "Full roll + flip"), i.e. the UniFuse loader's range, of which PanoFormer's loader is the quarter-circle variant. The literal PanoFormer range (`[0, W/4]`) is **not** implemented. Disclosed to Yixun in the reply to this query; the mirror operator and the frame audit are identical under either reading, so a correction before the FULL launch would only change the roll range.
2. "exp_36 … 就是 exp_43 的没有 flip 的版本" — as planned, exp_36 evaluates the *existing* continuous-YAW checkpoint (exp_15 / exp_17 arm `YAW`, the configuration that differs from `RF` only by the flip) at the C4 angles; it trains no model. exp_43's Stage B already contains the C4 block (30 cells) for `RF`. Told to Yixun in the reply.
3. The approval covers the plan's local commits on `exp-43-std-aug` (§6); nothing is pushed and no result is published without Yixun's go.

**Why:** the reviewer-facing augmentation baseline should be a recognised standard panorama augmentation; Yixun wants this row first.

## Query 6 — 2026-11-13 ~01:19 EST (button answer to "Start the exp_43 40k training after Codex code round 1 and review the eval kit while it trains?")

**Verbatim:**

> Yes, train early

**Summary:** the 40k training may launch after Codex code round 1, its fixes, and the gates L1–L5; the evaluation kit and chain are written and reviewed (rounds 2–3) during the 21.6 h run.

**Assumption / hypothesis:** the approval covers the review *order* only — not skipping any gate of plan §8, not a fourth review round, and not a push. The restart risk (a later round finding a training-path defect) was stated in the question's context and is accepted; it is written into plan §15 A1.

**Why:** saves ≈ 4–6 h on the rebuttal row; exp_43 is the highest-priority experiment (Query 5).
