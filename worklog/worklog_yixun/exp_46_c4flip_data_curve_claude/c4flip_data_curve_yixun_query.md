# exp_46 c4flip_data_curve — Yixun's queries (verbatim) and how they were read

Logged 2026-11-17 by the neuronic coordinating session (Claude Opus 5.5). The queries were asked in that session while
it was answering about exp_45; the same entries (Queries 6–8) are in exp_45's `rf_data_curve_yixun_query.md`.

## Query 6 — 2026-11-16 ≈ 11:20 EST

> flip + C4 yaw augmentation的data efficiency的experiment跑了吗？25%, 50%, 75%

**Summary.** Has the data-efficiency experiment (25 / 50 / 75 % of AR) of C4 yaw augmentation + flip run?
**Answer given.** No; nothing was submitted. The experiment that existed (exp_45) uses exp_43's RF arm, whose roll is
the full circle (no `offset_step`), not C4. Asked: run RF as planned, or a true C4 + flip arm?

## Query 7 — 2026-11-17 ≈ 11:00 EST

> 我要 C4 + flip

**Summary.** The wanted arm is C4 + flip, not RF.
**Not assumed; put to Yixun as C1–C5** (definition of C4, new experiment vs exp_45, RF on hold, a 100 % anchor, the
exp_45 round-3 decisions).

## Query 8 — 2026-11-17 ≈ 11:30 EST (answer to "按推荐批准 C1–C5 吗？")

> 按推荐批准

**Read as:** C1–C5 approved as recommended.
- **C1** C4 = rolls uniform over {0°, 90°, 180°, 270°} (`offset_step = 128` at `img_w = 512`), then the mirror flip
  with p = ½ (D4). Not PanoFormer's quarter-range roll.
- **C2** a new experiment, exp_46, forked from exp_45's kit after its code round 3 fixes, with its own plan and 3 + 3
  Codex rounds.
- **C3** exp_45's RF 25 / 50 / 75 % runs on hold (not cancelled).
- **C4** exp_46 includes a 100 % C4 + flip run (4 runs: 25 / 50 / 75 / 100 %).
- **C5** exp_45 code round 3 closes with the Coder's fixes of HIGH-1 / MEDIUM-2 (no 4th round), the r2-6
  materialisation exception approved, deviation 1 kept.

**Why this reading:** the question that preceded it listed exactly C1–C5 with a recommendation each; "按推荐批准"
answers that list. It is not read as the go for anything else (e.g. the K3D rerun).

**Assumptions recorded:** the experiment number 46 is free (no `exp-46*` branch on `origin` on 2026-11-17; the A6000
session, which numbers exp_40–45, could not be asked directly from neuronic — if it already uses 46, exp_46 is
renamed before any push).
