# exp_19 — closing analysis (reliability + interpretation)

**Analysis by Claude Fable 5 (main session seat). Coder seat: Opus 5 subagent.
Reviews: OpenAI Codex `gpt-5.6-sol` xhigh — r1, r2, closure (all archived in
this folder). Closure verdict: zero blocking findings; no published number
requires retraction.**

## What was measured

Four HAA-finetune arms from AR-40k EMA inits, released recipe (1,000 steps,
16×accum4=eff64, lr 5e-6, VAE frozen, weights-only init), single training seed
(42), evaluated on the full HAA Base test split (1,282 items / 4 rooms) at
K∈{8,1}, 5 eval seeds, `--cond-autocast bf16`, per-arm conditioning protocol
(announcement 05), per-scene recording. Headline convention = paper style
(room-mean → cross-room mean, T60 excludes dampened); pooled tables archived
alongside. Plus a 10-point finetuning-steps curve (K=8, seed 42) for P1/BF/YAW.

## Findings (K=8, ckpt-1000, paper convention)

| Arm | T60↓ | C50↓ | EDT↓ | R@1↑ |
|---|---|---|---|---|
| Vanilla (P1) | 3.413 | 2.202 | 84.99 | **5.184** |
| **YNA (Yaw-Aug init, aug OFF in FT)** | **3.391** | **2.096** | **77.26** | 4.761 |
| Yaw-Aug (aug ON in FT) | 4.092 | 2.777 | 91.76 | 4.133 |
| FA (B-F) | 4.892 | 3.210 | 113.03 | 3.959 |

1. **Sim2real inverts the AR ordering** (AR-40k: Yaw-Aug beat P1 on T60 both K;
   HAA: vanilla beats both invariance arms 3-for-3 among {P1,YAW,BF}).
2. **The YNA ablation attributes the inversion**: with the SAME Yaw-Aug
   representation but augmentation off during finetuning, acoustics match or
   beat vanilla (T60 −0.02, C50 −0.11, EDT −7.7, FD −0.005), retrieval slightly
   behind (R@1 −0.42). The transfer tax was the in-flight augmentation on a
   fixed-orientation 12-samples/room domain, NOT the representation.
3. **FA's tax is architectural and does not amortize**: the steps curve shows a
   plateau from ~410–500 (T60 4.7–4.9) with no further descent, while Yaw-Aug
   descends monotonically through 1000 (crossing below FA at ~500). The C4
   conditioning average dilutes the single-orientation signal at train AND eval
   and cannot be switched off.
4. **Program narrative (exp_17 + exp_19):** invariance-by-data buys 7–19×
   rotational flatness in-domain at zero inference cost (exp_17), and its
   representation transfers to a canonical-frame real domain at no acoustic
   cost once the augmentation is disabled for adaptation (YNA). Invariance-by-
   architecture buys exactness on the orbit but pays a permanent transfer tax.

## Reliability

- **Training seeds: one per arm.** All cross-arm deltas are conditional on
  seed 42; the T60 gaps at the endpoint (0.02–1.5) vs 5-seed eval σ (0.01–0.09)
  mean only the YNA≈P1 equivalence claim is at risk from training-seed noise —
  the FA/YAW-on gaps are far outside any plausible seed band.
- **Eval seeds: 5 everywhere** (endpoint tables); curve is single-seed 42 and
  used for dynamics only, never for row claims.
- **The 410-step "optimum"** for P1 is a curve reading at one eval seed; the
  registered endpoint remains 1000 (plan B1). Both are published.
- **Protocol integrity:** every published record carries cond_method/
  cond_autocast asserted at aggregation; BF eval used fa_invariant + C4 orbit +
  fwd-cap 64 (mirroring exp_07/exp_10 convention); the aggregator reproduces
  all published tables from raw JSONs (verified byte-for-byte on the four-arm
  table; persisted as exp19_aggregate.py).
- **Infra events with no numeric impact** (all documented in the worklog):
  AGREE_HAA.pt symlink fix (stock config path vs on-disk name); BF's three OOM
  false starts (0 checkpoints each — no partial artifacts); the gate-lock tee
  fd-inheritance bug (serialized launches; bug filed); EXPECT_SHA refusals
  working as designed after HEAD moved.
- **Accepted debt (closure review, all non-blocking):** extras/rotseeds/curve
  drivers trust filenames on resume (records were validated at aggregation
  instead); aggregator lacks fail-closed n_samples/EMA-source checks; eval
  driver's "grid 60" label is stale for 4 arms; no YNA-specific guardtests.

## Standing items produced by this experiment

- **Yixun 2026-08-19: FA still needs to beat vanilla FLAC on HAA** (issue_report
  -2). Candidate avenues listed there; the FA-init-vanilla-finetune arm
  (mirroring YNA) is the most promising untried one.
- **RAF open question** (issue_report -3): no FLAC numbers exist on RAF; the
  YNA recipe is the natural first arm; needs an exp_20-scale data adapter.
