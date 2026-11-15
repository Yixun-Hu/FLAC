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

## Query 4 — 2026-11-14 ≈ 19:05–19:20 EST, in the neuronic coordinating session (logged by that session)

Asked in the neuronic session, in this order (verbatim):

> Panoformer-style的yaw-augmentation你现在可以从github remote上面看到吗？应该是exp_43

> 你可以来跑一下Panoformer-style的yaw-augmentation的25%, 50%, 75% data的实验吗

> "RF（全圆 roll + flip，即 exp_43 的 arm）"中的RF，全圆是什么意思？

> Panoformer-style应该就是C4加上flip，对吧

> 但是如果我们之前选的是C的话，或者说是RF的话，那其实就和我们最开始跑的yaw-augmentation的实验是一样的了？就是加了一个flip?

> 2. panoformer原来为什么只是0°–90° 之间随机取一个角度？如果按照这个角度选择，放在我们orbitrir的故事里也是make sense的吗

> 1. UniFuse论文是什么论文？ERP是什么意思？
> 2. 我们现在是希望yaw-augmentation baseline弱一些，其实就是C4+flip或者是C4的yaw-augmentation就可以了，现在我们已经有C4或者是C4+flip的结果了嘛？

> 只跑 C4

> 另外25, 50, 75% data对应的实验应该都push上github了，你从remote能够看到： git fetch origin exp-45-rf-data-curve, then read worklog/worklog_yixun/exp_45_rf_data_curve_claude/rf_data_curve_brief.md and rf_data_curve_yixun_query.md on that branch. Follow the SOP: write plan_rf_data_curve.md for exp_45 and bring it to me for approval before implementing anything. Also push the exp_38 results commit to origin/exp-38-data-efficiency (D7-A in the brief).

(The last message also carried, pasted, the A6000 session's report of 19:13 EST: exp_43 Stage A result table, Stage B launch, the exp_45 brief.)

**Summary.** Yixun first asked whether the neuronic session could run the PanoFormer-style fraction runs, then
asked what "full circle" means, whether PanoFormer is C4 + flip (it is not: `randint(0, w//4)`, a random 0–90°
roll, + flip p = ½), why PanoFormer uses 0–90° (neither PanoFormer's paper nor its code gives a reason; UniFuse's
paper says only that ERP is invariant under yaw rotation), and whether a weaker C4 / C4 + flip baseline already
exists (no: only YAW, C16, RF and, as evaluation only, YAW at C4 angles). He answered "只跑 C4" to "C4 or C4 +
flip?", then sent the last message, which directs this session to the RF brief.

**Read as:** the last message supersedes "只跑 C4": exp_45 is the RF fraction curve of the brief, decided as the
brief records. No C4 training is planned; if a C4 data curve is still wanted it is a separate experiment
(it would need a 100 % C4 run too — none exists). This reading is stated to Yixun with the plan.

**Assumption, and why:** the SOP applies (plan → Codex plan review → Yixun's approval → implementation), as the
message says. D7-A cannot be executed as worded: there is no exp_38 results commit (plan F10, §10); what to push is
put to Yixun as decision N7 instead of being guessed.

## Query 5 — 2026-11-14 ≈ 20:55 EST (answer to "按推荐批准 exp_45 plan 和 N1–N10 吗？", neuronic session)

> 按推荐批准

**Read as:** plan r3 (`plan_rf_data_curve.md`) APPROVED with N1–N10 as recommended (§11): merge + kit fork; three
concurrent FULLs; `--logger none` + PL CSV; A6000 evaluation (tree 45) with the ten-cell RF-100 bridge; pushes of
`exp-45-rf-data-curve` each on Yixun's go; D7-A = a records-only exp_38 commit now (no preliminary numbers), formal
results after exp_38 step 4; the A6000 Claude session runs the A6000 / NAS steps; E0 before SMOKE; E0 cells exempt
from waveform dumps. The original instruction "push the exp_38 results commit to origin/exp-38-data-efficiency" plus
N7 is read as the go to push that records-only commit.
