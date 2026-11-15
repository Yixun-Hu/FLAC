# exp_45 rf_data_curve — Yixun's queries (verbatim) and how they were read

Logged 2026-11-14 ≈ 19:10 EST by the A6000 rebuttal session (Claude Opus 5.5, worktrees `exp-40` … `exp-44`
on `mae-cab-lab-server`). The queries were asked in that session; they are copied here so the neuronic session
that plans, implements and runs exp_45 has them in the repository.

## Query 1 — 2026-11-14 ≈ 12:30 EST

> PanoFormer-style的standard yaw-augmentation的data efficiency (25%, 50%, 75% data) 是不是应该可以放在neuronic上面跑了？

**Summary.** Run the data-efficiency points (25 / 50 / 75 % of AR) of the PanoFormer-style standard yaw
augmentation (exp_43's arm `RF`: full-circle roll + horizontal mirror flip, p = ½) on neuronic.

**Answer given (recommendation, five points):**

1. D8: the 25 / 50 / 75 % rows use RF instead of C16. The C16 fraction runs (exp_42) are put on hold, not
   cancelled; whether they are still wanted is decided once the RF fraction results exist.
2. D1 → neuronic: 2 × L40 × 32/GPU, the recipe of the exp_38 fraction runs (40,000 steps, seed 42, exp_38's
   subsets). The evaluation protocol is exactly exp_43 Stage A's (full 6,337-item unseen split, θ = 0,
   K ∈ {1, 8}, eval seeds 42–46).
3. A new experiment, exp_45, with its own plan and Codex reviews. The A6000 session writes a brief (source
   commits, recipe, evaluation protocol, acceptance criteria); the neuronic session writes the plan, and after
   Yixun approves it implements and runs it.
4. Preparation starts now; the FULL trainings start only after the RF 100 % θ = 0 result is confirmed.
5. The neuronic session pushes exp_38's results commit (exp_42 decision D7-A), so the YAW / OrbitRIR fraction
   rows enter the comparison from a hash-pinned commit instead of a pasted table.

## Query 2 — 2026-11-14 ≈ 19:00 EST (answer to "按推荐方案让 neuronic 跑 RF 的 25/50/75% 吗？")

> 按推荐做

**Read as:** all five points above approved.

## Query 3 — 2026-11-14 ≈ 19:03 EST

> 我这边是不是直接让neuronic的claude session来跑实验就可以了？

**Read as:** Yixun will hand the experiment to the neuronic Claude session. That session plans, implements and
runs exp_45 itself; the A6000 session has no neuronic access and only supplies the brief
(`rf_data_curve_brief.md`, this folder).

**Assumption, and why:** the SOP still applies on neuronic (plan approved by Yixun before implementation,
TDD, Opus 5.5 subagents for code, Codex ≤ 3 plan + 3 code rounds), because "直接让 … 跑" says who runs it,
not that the SOP is waived.
