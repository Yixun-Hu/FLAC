# Lab notebook — exp_14_fa_drawshare

## 2026-08-12 — scaffold + plan Rev 2
- **Goal** — does the operational chunk plan (per-angle vs fully-shared RoPE draws) change FA training outcomes at 40k? Single delta, seed 42, sequential arms.
- **Decisions on record** — Yixun approved the discriminating training ("做，要把因果钉死"), then, after the plan review showed 12 days cannot support a general causal claim, chose **option B** (run the scaled-down version, scope the claim to the seed-42 trajectory). Sequencing "顺序跑".
- **Rev 2 applied all review findings**: cap 32 vs **96** (1/3 vs **3/3**, matching exp_11 — the earlier "micro-32 tops out at 2/3" was my arithmetic error); config key instead of env override; from-scratch launcher; claim scoping; DS2 downgraded to a cross-era replication check; registered trajectory statistic; fit probe gate for cap 96.
- **Result** — `launched` (planning). Re-review → Yixun's final go → fit probe → DS-PA → DS-CS3.
