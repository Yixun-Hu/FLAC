# Analysis — exp_08_fa_matched (reliability + exp_07 decision framing)

**Analyst:** Opus 4.8 (main session, max effort).
**Role-transfer flag (per Yixun 2026-07-09):** the SOP routes the "reliability + next-step analysis" to the main-session model. Historically that was Fable 5; after the `/model` switch it is **Opus 4.8**. This document — the interpretation, the reliability judgement, and the exp_07 recommendation — is **Opus-authored**, not Fable's. I flag it because a next-step recommendation is a judgement call and its provenance (which model made it) should be transparent when the analyst identity changes mid-experiment.

---

## 1. What exp_08 actually establishes

exp_03–06 left one thing unmeasured: at a *matched* recipe and a *fully-characterized* fine-tune operating point, what is frame averaging's own marginal effect — separated from the released-checkpoint lineage problem that blocked exp_03–06. exp_08 answers exactly that by holding everything constant except `--cond-method` and reading A-F − A-V.

Three findings, in decreasing order of confidence:

1. **Exact C₄ invariance is real and cheap (H-A2 + H-A3, PASS/PASS).** relL2 ≈ 0.0023 across the C₄ subgroup — ~90× below the vanilla yaw gap and under the pre-registered bf16 floor — and the downstream acoustic metrics are flat well inside the 2σ noise band (C₄ flatness ranges 20–185× smaller than the corresponding 2σ single-eval noise). This reproduces exp_03's conditioning-level proof (4.9e-8) now *end-to-end through a trained generator on the full split*. The minimum project goal is met on a trained model. This is the robust core of the experiment.

2. **The T60 gain is real and seed-robust (H-A1 T60 cell + M5).** A-F beats its matched vanilla control on T60 by −0.28 (K=1) / −0.44 (K=8), recovering 41% / 59% of the fine-tune's T60 damage. M5 confirms the effect survives a training-seed change (−0.46 at s42, −0.35 at s43; worst per-arm seed swing 0.19 < half the effect 0.23). This is the first intervention across six experiments that pushes a fine-tuned model *back toward* the released baseline on the headline metric.

3. **The K=8 EDT/C50 costs are not seed-resolvable; the K=1 costs stand (H-A1 EDT/C50 cells).** At seed-42 5-seed bands, A-F regresses EDT and C50 at *both* K, each many σ_c outside the equivalence band — hence the strict H-A1 FAIL (6/6 T60/C50/EDT cells outside 2σ_c: 2 superior T60, 4 regression). M5's downgrade rule then fires on the **K=8** EDT and C50 cells (training-seed swing 0.64 / 0.019 exceeds half the FA effect), so those two are not cleanly attributable to frame averaging. But **M5 was a K=8 screen only** — the **K=1** EDT (+1.02, +7.3σ_c) and C50 (+0.023, +2.5σ_c) regressions were never retrained at a second seed and remain strict, un-downgraded regressions.

**Net:** the training-seed-robust marginal effects of `fa_invariant` at this operating point are a positive one (T60, both K) and — only at K=8 — the dissolution of the early-field costs into seed noise. **I do not claim non-inferiority:** the strict H-A1 gate FAILs and the K=1 early-field regressions are real and untested for seed sensitivity. The honest closure is *strict H-A1 FAIL; T60 superior and K=8-seed-robust; K=8 EDT/C50 seed-indeterminate; K=1 EDT/C50 remain strict regressions* — recorded as such, without retroactively moving the pre-registered line.

---

## 2. Why this trade shape? (mechanistic hypothesis — consistent with, not proven by, the data)

The direction of the effects is physically coherent:

- **T60 is a late-field, whole-room decay statistic** — governed by volume and total absorption, which are *yaw-invariant* by construction. Averaging the geometry embedding over 4 yaw rotations regularizes it toward that rotation-invariant quantity, reducing variance in exactly the direction the physics says is correct. An invariant conditioner *should* help a rotation-invariant target. This is the mechanism I'd expect a priori, and it is what we see (and it is seed-robust).
- **C50 and EDT are early-field statistics** — direct path + first reflections, which *are* orientation-sensitive (which wall faces the listener). Frame-averaging over yaw could mildly blur direction-specific early-reflection cues. This is a plausible mechanism for a small early-field cost — and *at K=8* M5 puts that cost at or below the training-seed noise floor, so if the mechanism is real it is small there and not established. At **K=1** the cost was not seed-tested and remains a strict regression, so the mechanism cannot be dismissed at K=1.

I state this as a hypothesis to be tested by from-scratch training, not a conclusion. The clean prediction it makes: from-scratch, where the geometry embedding is *learned* under the averaging rather than inheriting a released embedding, the early-field cost (if mechanistic) should stay small or vanish while the T60 benefit persists.

---

## 3. Reliability analysis — threats to validity

**Controlled:**
- *Eval-precision confound* — the M1.5 bf16 mirror is the comparator (bf16 shifts A-V T60 +0.12; using the fp16 row would have flattered A-F). Both arms read at `--cond-autocast bf16`.
- *Training-seed confound* — M5 pair + pre-registered downgrade rule (§3 of results).
- *Control-reuse legality* — V1′ reused as recipe-equivalent (code-diff proof: `5d1c64c..HEAD` on `finetune_cond.py` is flag-gated no-ops for a constant-lr/warmup-0/freeze-bn run); recorded per the plan review's SOP conditions. Not claimed bit-identical.
- *Implementation regression* — H-A2 exactness (relL2 ≈ floor, not ≈ vanilla gap) rules out a broken FA path; consistent with exp_03 cycle-4 proofs.
- *Aggregation* — every headline number flows from `aggregate_results.py` over committed JSONs, not hand transcription.

**NOT controlled (and why it's acceptable here):**
- **Operating point is a damaged fine-tune, not the released model.** A-F and A-V both sit well below the released baseline (A-F K=8 T60 8.916 vs 8.609). exp_08 is a *matched marginal* comparison at that damaged point, by design — it deliberately sidesteps the exp_03–06 lineage blocker. The marginal FA effect measured here **may not transfer** to a from-scratch/undamaged model. This is the single largest external-validity gap and is precisely the exp_07 question.
- **Short fine-tune (625 opt steps).** The measured effect is an *early-fine-tune* marginal effect; a longer schedule could shift it. Accepted because the matched comparison is internally valid regardless of horizon.
- **C₄-only invariance.** 45° is not covered (relL2 0.206) — structural (patch tokens aren't roll-equivariant off-group), pre-registered as a C₄ guarantee. A continuous-yaw guarantee would need a different construction (e.g. steerable features or a denser group), out of scope.

No result here depends on an uncontrolled factor in a way that changes its verdict: H-A2/H-A3 are implementation facts; the T60 gain is seed-robust; the **K=8** EDT/C50 costs are explicitly downgraded to indeterminate (not asserted as FA-caused), while the **K=1** EDT/C50 regressions are reported as standing strict regressions rather than explained away.

---

## 4. The exp_07 decision (framed for Yixun — this is the standing "after exp_08, decide" call)

exp_08 sharpened the from-scratch question into two concrete, pre-registerable sub-questions:

1. **Lineage:** does from-scratch *vanilla* (B-V) at paper scale (67.5k steps, the count found in `FLAC.ckpt`) reach the released Table-1 numbers? If yes, the lineage gap that blocked exp_03–06 is data/optimizer-state, not architecture, and the released operating point becomes reachable. If no, the residual gap is quantified before any month-scale FA run.
2. **Does the trade survive from-scratch?** does from-scratch `fa_invariant` (B-F) keep the T60 gain and exact invariance while the EDT/C50 cost stays small/vanishes (the §2 prediction)? Only from-scratch removes the "repairing fine-tune damage" confound and can address the **maximum goal** (beat released Table-1 K=1/K=8).

**What exp_08 does and does not license about exp_07:**
- It does *not* prove exp_07 will succeed — the marginal effect is measured at a damaged point (§3).
- It *does* de-risk exp_07: FA is exactly invariant end-to-end and adds a seed-robust T60 gain at both K. Its early-field costs are real at seed-42 bands; at K=8 they are seed-indeterminate, at K=1 they are untested and stand. There is no evidence that FA is *intrinsically* harmful (the pre-registered "FA materially worse → exp_07 changes character" branch did **not** trigger — T60 improves and the costs are small/indeterminate), but the accuracy trade is not fully discharged and remains an open question for from-scratch training.

**Cost (from the exp_07 plan, review-anchored):** paper-parity is ~10 d (B-V) + ~30 d (B-F, 4× conditioner cost) ≈ **40 GPU-days sequential**; hybrid (c) spends the first ~10 d on B-V (with 10k-step screening evals) to resolve sub-question 1 *before* committing ~30 d to B-F.

**New, relevant observation (GPU state, 2026-07-09):** GPU 0 is already running a from-scratch **vanilla** FLAC training in your `rir2rir-oneroom` workspace (`FLAC_vanilla291k`, `FLAC_AR_noAGREE.json`, eff-batch 64, 291k-step target, 1 d 6 h in). I do not know its purpose and have not touched it, but it bears on the decision two ways: (a) it is *evidence in progress* on "does from-scratch vanilla FLAC train sanely" (albeit at a different recipe than exp_07's matched B-V — eff-batch 64 vs 128, 291k vs 67.5k, noAGREE), and (b) it is **GPU contention** — both GPUs are currently occupied by your other jobs, so exp_07 cannot launch today without freeing one. Worth reconciling exp_07's B-V arm against that existing run before paying to repeat it.

---

## 5. Opus recommendation

Given exp_08's outcome, my recommendation is **exp_07 hybrid (c), but sequence-aware of the GPU-0 run**:

1. **First decide whether the in-flight `FLAC_vanilla291k` can serve as (or seed) the B-V lineage probe.** If its config/recipe is close enough to serve as the vanilla reference, exp_07 may only need the B-F arm + a matched-recipe B-V spot-check, cutting ~10 d off the budget. If it's for a different purpose, keep exp_07's own matched B-V.
2. **Run B-V (or reuse) to a 10k-step screening cadence** and check the pre-registered rule: B-V trajectory heading within 2σ of released Table-1 ⇒ proceed to B-F at matched steps; materially short ⇒ stop and quantify the residual data/env gap before spending a month.
3. **Only then commit B-F.** The §2 prediction (T60 gain persists, early-field cost small) is the falsifiable target; exact invariance is already assured by construction.

This spends the first week resolving the biggest unknown (is the released operating point reachable from-scratch, and is the GPU-0 run already answering it) before committing the expensive FA month — and it keeps the maximum-goal question alive, which exp_08 by design could not touch.

**The budget/GPU-ownership call is yours** (weeks of GPU 1, or cluster/H100 access per exp_07 plan §2 option (a)); exp_08 has removed the scientific reason to hesitate, not the resource one.
