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

Both from random init (same seed policy), same steps, same ~~effective batch 128~~ **effective batch 64 (CORRECTED by the 2026-07-10 audit — paper §B.1 + defaults.ini + ckpt counter-proof; the 128 traced to the ckpt-incompatible README example)**, original optimizer config (lr 5e-5 + InverseLR, **EMA ON** — from-scratch is exactly where the original recipe belongs; no freeze-bn: BN must learn its own stats now). Two arms because exp_06 proved absolute comparability to the released numbers is not guaranteed — the **primary comparison is B-F vs B-V at matched lineage** (FA's true effect), with released-Table-1 numbers as context per announcement 01 (full-split evals, identical protocol).

**Code required: ONE TDD round (plan-review blocking finding 1):** `--max-steps` for `train.py` (currently hardcoded 1,000,000 with no override — a pre-registered budget cannot be enforced without it); tests: parser default preserves 1M, value reaches `pl.Trainer(max_steps=...)`. Resume needs NO code (`--ckpt-path` → `trainer.fit(ckpt_path=...)` exists); seed needs NO code (`--seed` exists). Otherwise: `train.py` already consumes `training.cond_method`/`frame_avg_angles` via `create_training_wrapper_from_config` (exp_03 cycle-4 plumbing, dispatch in all three step methods, ValueError on unknown). Arms differ by one config copy in this folder (FLAC_AR.json + `"cond_method": "fa_invariant"`). If the plan review finds any train.py gap (e.g. seed control, resume flag for a weeks-long run — see §4), it becomes one TDD round.

## 2. THE BUDGET DECISION (Yixun input required) — re-anchored by the plan review's discovery

**The released checkpoint records its own training length: `global_step = 67,500` (epoch 14, `FLAC.ckpt`).** Paper scale is therefore a fixed target, not speculation. Throughput anchors (A6000; EMA-off measurements — an EMA-ON fit/throughput probe runs before trusting these, per review finding 3): vanilla ≈ 10 samples/s free-GPU. ~~Table below assumed effective batch 128~~ **TABLE CORRECTED by the 2026-07-10 audit to the released effective batch 64** (4.32M samples at 67.5k steps):

| Budget | B-V wall-clock (GPU 1) | B-F wall-clock | Total (sequential) | What it supports |
|---|---|---|---|---|
| **67.5k (paper-parity)** | **~5.0 d** | **~16.7 d** | **~21.7 d** | full H3 + Table-1 comparison at matched steps |
| 33k (half) | ~2.5 d | ~8.2 d | ~10.7 d | H1/H2 + matched-budget H3; trajectory extrapolation |
| 10k (pilot) | ~0.75 d | ~2.5 d | ~3.2 d | pipeline validation + early trajectory only |

Ways forward (pick one):

- **(a) Cluster/multi-GPU hardware** — if you have access (~~the original used ≥2 GPUs~~ **CORRECTED by the 2026-07-10 audit: the original used a single H100, eff batch 64, accum 1** — the ≥2-GPU belief traced to the README example, which is ckpt-incompatible; multi-GPU would be OUR deviation, usable for speed but labeled): full-scale becomes days. The SOP's commit+push-before-remote rule applies; resume via existing `--ckpt-path` (review-verified, §4).
- **(b) Matched-budget reduced scale on GPU 1** — e.g. 50k steps both arms (~~1 month~~ **~16 d sequential at the corrected eff-64**; or interleave). Honest framing: "FA vs vanilla at matched budget," absolute numbers below paper scale; H1/H2 fully answerable, H3 answerable *relative to B-V* only. **Largely obsolete post-audit: paper-parity (a/c) now costs ~21.7 d total, barely more than this.**
- **(c) Hybrid (recommended, sharpened):** B-V straight to 67.5k (~~10 d~~ **~5 d, corrected**; 10k-step screening evals en route give the 'distance-to-Table-1 vs steps' curve). Decision rule, pre-registered: if B-V@67.5k lands within 2σ of the released numbers, lineage is vindicated and B-F runs to the SAME 67.5k (~~30 d~~ **~16.7 d, corrected**; or less if the conditioner-cache optimization is commissioned); if B-V@67.5k falls materially short, the residual lineage gap (data/env) is quantified BEFORE spending ~2.5 weeks on B-F, and we reconvene.

**Planner recommendation: (c)** — it spends the first ~5 days resolving the biggest unknown (whether the released numbers are reachable from-scratch on our data/env) before committing the ~17-day B-F arm.

## 3. Evaluation & acceptance (pre-registered)

- Checkpoint cadence: ~~every 10k steps~~ **every 2,500 steps (CORRECTED by the 2026-07-10 audit — matches the released run's recorded `ModelCheckpoint(every_n_train_steps=2500)`, enabling checkpoint-selection-curve mirroring; screening evals still at 10k-step marks)**; screening evals are EXTERNAL `eval_FLAC.py` jobs (train.py's --val-every logs denoising loss only — review finding 4), at K=8 seed 42: **EMA weights via the default eval path, online weights via a committed eval-config copy with `training.use_ema=false`** (both configs in this folder; clean-load asserts on).
- Final: full 5-seed × K∈{1,8} on both arms (announcement 01 protocol); H1 (Metric-1 ≡ 0 on C₄, comparator) + H2 (per-angle Metric-2 flatness) rotation sweeps on B-F; bf16 floor re-registered before reading.
- **Primary claims with pre-registered thresholds (review finding 5):** (i) **non-inferiority:** B-F within 2× the combined 5-seed σ of B-V on each of T60/C50/EDT at both K (R@k advisory); superiority claimed only if B-F beats B-V by >2σ_combined; (ii) B-F passes the cylindrical sanity check exactly (Metric-1 ≡ 0 on C₄ at the pre-registered bf16 floor; H2 flatness within 2× single-eval noise). **Stretch:** B-F vs released Table 1 at matched 67.5k steps. Seed policy: training seed 42 both arms; eval seeds 42–46. Headline = final full-protocol evals only; all 10k-cadence numbers are screening.
- Stop rules: divergence/NaN → infra-vs-bug triage per SOP; budget checkpoint at each 10k-eval — Yixun can stop/extend either arm.

## 4. Risks

- **Wall-clock dominates everything**; fa_invariant's 4× conditioner cost is the multiplier to watch (a conditioner-caching optimization — precompute the 4 rotated ViT features per sample — could cut B-F toward B-V cost, but that's new code + review + cache-correctness risk; deferred unless (b) is chosen and the month is unacceptable).
- Long-run interruptions: covered by train.py's existing `--ckpt-path` resume (review-verified; no new code); resume drills recorded in the notebook. EMA-on memory/throughput probe (10 steps, GPU 1) runs before any launch — README's H100 note applies to the VAE, not the DiT (CLAUDE.md: everything but VAE fits 24 GB), but we verify rather than assume.
- GPU 1 ownership: currently free, but it is the other user's usual slot — Yixun should confirm it can be occupied for weeks.
- VAE: reused frozen from the release (`--pretransform-ckpt-path weights/FLAC/VAE.safetensors`) — pretrained VAE is not part of the lineage problem (eval-only component, exp_01 reproduced through it).

---

## REVISION 2026-07-10 (Fable 5) — config-identity audit corrects the recipe anchor

Per Yixun's pre-launch ask, `fa_scratch_config_identity_audit.md` cross-checked paper text, repo configs, current-branch code, and **the released ckpt's internal records**. Two corrections to this plan:

1. **Effective batch is 64, not 128** (paper §B.1 "batch size of 64 on a single H100"; `defaults.ini` 64×1×1; ckpt loop counters prove accum=1.0000 and 4,550 steps/epoch = floor(291,210/64)). §2's wall-clock table halves: **B-V ~5 d, B-F ~16.7 d, sequential ~21.7 d** at 67,500 steps (= the released ckpt's recorded budget; a periodic 2,500-cadence checkpoint, so checkpoint-selection curves are mirrorable).
2. **Arms are built and verified:** `FLAC_AR_BV.json` (byte-copy) / `FLAC_AR_BF.json` (+`cond_method`,`frame_avg_angles` only); instantiation asserts pass (identical 64.50M architecture; wiring per recipe). Micro-batch on the 48GB A6000 may force micro×accum ≠ 64×1 — pre-registered rule: keep eff 64, same micro×accum in BOTH arms; M0 fit probe decides.

Hybrid (c) structure unchanged. Reviewer for all exp_07 rounds: Codex **`gpt-5.6-sol` xhigh** (Yixun 2026-07-10; CLI upgraded to 0.144.1).
