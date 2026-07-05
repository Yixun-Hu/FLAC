# Lab notebook — exp_03_fa_invariant_cond

## 2026-07-04T22:10:15-04:00 — scaffold + plan drafted
- **Goal** — Route 1 (hard invariant conditioning: C4 frame averaging over DINOv3 path + cylindrical pose invariants), per Yixun's Query 1. TDD, small commits.
- **Hypothesis** — symmetrized conditioning gives Metric-1 ≡ 0 on C4 by construction; after non-destructive fine-tune, Metric 2 at α=0 within ~2σ of exp_01.
- **Change** — worklog scaffold only (query, plan, this notebook). No source code touched.
- **Version Control** — branch check-equivariance-necessity, base_commit 9dabe7b (TDD SOP commit; code base unchanged since 0bd5da0 + exp worklogs).
- **Result** — `launched` (planning phase); plan awaiting Yixun approval before any TDD cycle starts.
- **Analysis** — key design decisions needing sign-off: 3-dim (r,z,Δφ) pose encoding (warm-start-friendly) vs 4-dim; scope = single new method name fa_invariant; R1 vanilla-control gate before reading R2.
- **Next** — on approval: TDD cycle 1 (conftest + cylindrical tests RED).
