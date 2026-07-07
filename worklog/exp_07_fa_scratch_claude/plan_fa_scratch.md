# Plan — exp_07_fa_scratch (Route B: from-scratch fa_invariant training)

**Author:** Fable 5 (Planner) · **Coder:** Opus 4.8 max (config-only; TDD round only if code proves necessary) · **Reviewer:** Codex gpt-5.5 xhigh · **Date:** 2026-07-07
**Status:** AWAITING plan review + Yixun approval (this plan contains a BUDGET DECISION only Yixun can make).

## 0. Why from-scratch, and what it buys

exp_03–06 closed every recipe-space explanation for the fine-tune blocker; the released checkpoint is not reachable-from nor returnable-to under the shipped objective+data. From-scratch training removes every lineage confound at once (optimizer state owned from step 0 — Yixun's hypothesis moot; data/env lineage identical across arms by construction) and is the only evidence-supported route to H3-grade accuracy claims and the Table-1 (maximum) goal.

## 1. Design — two matched arms (the control is non-negotiable)

| Arm | Conditioning | Everything else |
|---|---|---|
| **B-V** (control) | vanilla | identical |
| **B-F** (method) | `fa_invariant` (C₄ FA + cylindrical poses) | identical |

Both from random init (same seed policy), same steps, same effective batch 128 (original recipe), original optimizer config (lr 5e-5 + InverseLR, **EMA ON** — from-scratch is exactly where the original recipe belongs; no freeze-bn: BN must learn its own stats now). Two arms because exp_06 proved absolute comparability to the released numbers is not guaranteed — the **primary comparison is B-F vs B-V at matched lineage** (FA's true effect), with released-Table-1 numbers as context per announcement 01 (full-split evals, identical protocol).

**Code required: none expected.** `train.py` already consumes `training.cond_method`/`frame_avg_angles` via `create_training_wrapper_from_config` (exp_03 cycle-4 plumbing, dispatch in all three step methods, ValueError on unknown). Arms differ by one config copy in this folder (FLAC_AR.json + `"cond_method": "fa_invariant"`). If the plan review finds any train.py gap (e.g. seed control, resume flag for a weeks-long run — see §4), it becomes one TDD round.

## 2. THE BUDGET DECISION (Yixun input required)

Measured throughput anchors (A6000, this repo): vanilla ≈ 7.8 samples/s shared → ~10 samples/s on the free GPU 1; fa_invariant ≈ 0.3× of vanilla (4× ViT conditioner cost) → ~3 samples/s. At effective batch 128:

| Steps | B-V wall-clock (GPU 1) | B-F wall-clock | Total (sequential) | What it supports |
|---|---|---|---|---|
| 50k | ~7 d | ~24 d | ~31 d | probably under-trained; H1/H2 yes, weak H3 |
| 100k | ~15 d | ~48 d | ~63 d | ? — original step count unknown (train.py ceiling 1M) |
| 200k+ | ~30 d | ~96 d | months | approaching paper scale |

**On this box, from-scratch at paper scale is a months-long commitment.** Three ways forward (pick one):

- **(a) Cluster/multi-GPU hardware** — if you have access (the original used ≥2 GPUs; H100s would be ~3–5× faster per GPU): full-scale becomes 1–3 weeks. The SOP's commit+push-before-remote rule applies; I'd add a small `--resume-from` TDD round (mandatory for any weeks-long run regardless of venue).
- **(b) Matched-budget reduced scale on GPU 1** — e.g. 50k steps both arms (~1 month sequential; or interleave by alternating days). Honest framing: "FA vs vanilla at matched budget," absolute numbers below paper scale; H1/H2 fully answerable, H3 answerable *relative to B-V* only.
- **(c) Hybrid** — B-V at 50k first (validates the from-scratch pipeline + gives the control), decide B-F scale after seeing B-V's curve vs Table 1 (if B-V@50k is already close to paper numbers, the original likely trained ≪1M steps and (b) suffices).

**Planner recommendation: (c)** — it spends the first week resolving the biggest unknown (how many steps the original actually needed) before committing months.

## 3. Evaluation & acceptance (pre-registered)

- Checkpoint cadence: every 10k steps, EMA + online both evaluated at K=8 seed 42 (screening trajectory — "Table-1 distance vs steps" curve).
- Final: full 5-seed × K∈{1,8} on both arms (announcement 01 protocol); H1 (Metric-1 ≡ 0 on C₄, comparator) + H2 (per-angle Metric-2 flatness) rotation sweeps on B-F; bf16 floor re-registered before reading.
- **Primary claims:** (i) B-F ≈ B-V on T60/C50/EDT/R@k at matched budget (FA costs nothing) — or better; (ii) B-F passes the cylindrical sanity check exactly (minimum project goal, on a trained-from-scratch model). **Stretch:** B-F vs released Table 1 (maximum goal — only meaningful at scale (a)).
- Stop rules: divergence/NaN → infra-vs-bug triage per SOP; budget checkpoint at each 10k-eval — Yixun can stop/extend either arm.

## 4. Risks

- **Wall-clock dominates everything**; fa_invariant's 4× conditioner cost is the multiplier to watch (a conditioner-caching optimization — precompute the 4 rotated ViT features per sample — could cut B-F toward B-V cost, but that's new code + review + cache-correctness risk; deferred unless (b) is chosen and the month is unacceptable).
- Long-run interruptions: `--resume-from` for train.py-style runs (Lightning ckpt_path) — small TDD round before launch, mandatory for multi-week runs.
- GPU 1 ownership: currently free, but it is the other user's usual slot — Yixun should confirm it can be occupied for weeks.
- VAE: reused frozen from the release (`--pretransform-ckpt-path weights/FLAC/VAE.safetensors`) — pretrained VAE is not part of the lineage problem (eval-only component, exp_01 reproduced through it).
