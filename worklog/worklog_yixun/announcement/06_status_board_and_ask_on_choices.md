# 06 — Status board every response; ask on every real choice (standing, Yixun 2026-08-11)

## Original instruction (verbatim)

> Everytime you respond please tell me what are all the status for experiment and their ETA (ETD time) and the earlist time you need my intervention or deision to steer you. And everytime you encounter problem that you want me for confirmation about the choices or how to do/waht to do, you need to ask me.

## Rule

1. **Every response ends with a status board** covering ALL live experiment work — every running/queued/blocked item (Slurm jobs, subagent rounds, reviews, analyses in flight) with: current state, wall-clock ETA/ETD (not just duration), and **the earliest moment Yixun's intervention or decision could be needed** ("no presence needed until X" when fully autonomous). This strengthens the CLAUDE.md wait-time mandate from "when runs are in flight" to *every response*.
2. **Every genuine choice point is asked, not decided.** When a problem arises where Yixun's confirmation of the choice, method, or scope could change the path (hardware substitutions, protocol deviations, budget/queue tradeoffs, retraction/rerun decisions, anything not already pre-approved in an approved plan), STOP and ask with concrete options before proceeding. Pre-approved plan steps and reversible mechanical work continue autonomously.

## Why

Yixun steers multiple concurrent experiments across machines; the status board is the steering interface, and silent unilateral choices at fork points cost more to unwind than the pause to ask.
