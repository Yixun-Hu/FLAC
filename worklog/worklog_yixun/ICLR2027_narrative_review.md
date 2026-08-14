# ICLR 2027 narrative review — "The m=0 Problem"

**Author:** main session, Opus 5 (max effort) — 2026-08-14.
**Role note per CLAUDE.md:** this is an *analysis* artifact, a Fable 5 seat, authored by Opus 5 after a mid-session model change. Flagged as required.

**Scope of the read:** every experiment record in `FLAC/worklog/worklog_yixun/` (exp_01–exp_16, plus `model_comparison.md`, `HANDOFF.md`, `issue_report.md`, `master_experiment_tracker.md`) and in `cylindrical-dinov3/worklog/` (both `worklog_yixun/` exp_01–exp_12 and `worklog_yixun_neuronic/` exp_01–exp_06), plus a literature novelty check.

**Question asked (Yixun, 2026-08-14):** what should the paper narrative be for ICLR 2027 (abstract 18 Sept, paper 25 Sept)? Specifically — (1) are there publishable points beyond "FA beats vanilla at 40k"; (2) can the FA result alone carry an ICLR paper?

**Answer in one line:** No to (2), and yes to (1) — the strongest paper in the corpus is the azimuthal-harmonic account buried in `cylindrical-dinov3/worklog/worklog_yixun/PHYSICS_CONSTRAINTS_synthesis_exp05_to_exp10.md`.

---

## 1. Why the FA story cannot carry the paper

Each item below is already in our own records, so it would have to appear in the paper's own ablations. A reviewer will find them.

### 1.1 The same single-delta reverses sign under a better recipe

Matched steps, matched seed, K=8, FA vs its vanilla twin:

| Recipe · arm @40k, K=8 | T60 ↓ | C50 ↓ | EDT ↓ | R@1 ↑ |
|---|---|---|---|---|
| Legacy 2-GPU — vanilla (P1) | 8.993 | 1.0093 | 40.650 | 5.173 |
| Legacy 2-GPU — FA (B-F) | **8.202** | **0.9778** | **38.793** | **5.387** |
| exp_11 L40 — vanilla (VANL) | **8.048** | 1.0219 | **37.319** | 4.949 |
| exp_11 L40 — FA (C4L) | 8.414 | **1.0095** | 41.499 | 5.091 |

Seed-paired delta under the exp_11 recipe: **T60 +0.366 ± 0.011, EDT +4.180 ± 0.091 — FA is worse.**

The mechanism matters: the better recipe improved *vanilla* by 0.945 T60, which is larger than FA's entire claimed advantage. This is the classic "gain disappears against a properly tuned baseline" pattern.

### 1.2 The 40k win point is a checkpoint-band artifact

`InverseLR` holds lr ≈ 4.8e-5 for the whole run, so adjacent checkpoints swing ~±0.5 T60. FA@40k reads 8.190 while *every* neighbour on both sides is 0.7–1.5 higher: 30k 8.858, 32.5k 9.022, 35k 9.733, 37.5k 9.176. The exp_10 A3 diagnostic already downgraded "FA wins 12/14 cells" to **band parity**.

### 1.3 At matched compute FA is not competitive

FA costs **3.5× per step**. Equalising compute against the 87.5k vanilla anchor puts FA at 25k steps:

| | T60 | C50 | EDT | R@1 |
|---|---|---|---|---|
| FA @25k (≈ anchor compute) | 8.316 | 1.0772 | 40.720 | 4.393 |
| vanilla anchor @87.5k | 8.293 | 0.9660 | 35.951 | 6.959 |

T60 ties; everything else loses badly.

### 1.4 The best model in the program is vanilla

P1@87.5k beats released Table-1 on 8/8 cells at both K. It is batch-norm statistics, DDP and a longer budget — nothing to do with equivariance.

### 1.5 Every arm is a single training seed

Five *eval* seeds control eval noise, not run-to-run training variance. The FA effect (~0.4 T60) is smaller than the spread between our own three vanilla @40k controls (8.048 / 8.611 / 8.993).

### 1.6 The invariance FA delivers is not the invariance that matters

exp_14 measures C₄'s degradation under uniform random yaw as statistically indistinguishable from vanilla's (p ≈ 0.5). Exact invariance at four angles buys nothing on the continuous distribution.

---

## 2. The experiment that would rescue it will not report in time

exp_14 `fa_drawshare` (DS-PA, running now) is the causal test of the reversal. exp_11's analysis names the prime suspect: the batched-orbit path **shares one stochastic RoPE rescale draw per chunk** where the legacy loop drew per angle, reducing augmentation diversity *only* on the FA side. Two readings, opposite implications:

- **Implementation artifact** — chunk-shared draws crippled FA training; the FA advantage is real and exp_11's C4L was handicapped.
- **Genuine recipe interaction** — FA's advantage exists only against a weak baseline; the FA headline is dead.

**Timing:** DS-PA is at ~4,400 / 40,000 steps after 22 h (~11%), running ~198 steps/h under co-tenancy → **~22 Aug**. DS-CS3 is gated behind it by design → **~1 Sept** before any evaluation. Too late to build on, and it might not go our way.

---

## 3. The binding constraint — compute does not fit, and priorities are inverted

| Work | Pool | State | Lands |
|---|---|---|---|
| exp_14 DS-PA | A6000 ×2 | ~11% of 40k | ~22 Aug |
| exp_14 DS-CS3 | A6000 ×2 | gated, not started | ~1 Sept |
| exp_15 yaw augmentation | cluster L40 | leg 1 of 16, launched 14 Aug 11:20 | queue-dependent |
| exp_16 ARE-V | A6000 ×2 | **blocked** — plan REQUEST-CHANGES | not scheduled |
| exp_11 100k orbit legs | cluster L40 | 4 of 5 never started | no horizon |

Two independent pools. The 8×L40 cluster trains a 40k arm in ~13 h of compute but is queue-bound (hence exp_15's 2,500-step chained legs). The A6000 box is uncontested but slow, and hosts the two longest jobs.

**The inversion:** eight GPU-days are committed to nailing down a foil, another eight to a method whose plan review returned four blockers, while the one experiment that would make this a positive-result paper has not been designed.

### 3.1 exp_16 ARE is not two days from running

Codex plan review returned **REQUEST-CHANGES, 4 blockers**, at least two substantive:

- Changing the training target without also changing the forward-process `x_t` construction would train an **inconsistent field**.
- P1 is **not** a bit-identical λ=0 control — same-seed P1 reruns already differ materially (K=8 EDT 35.955 vs 38.845 at 87.5k). A contemporaneous λ=0 arm is needed for any causal claim.
- The anchor's VAE encode is stochastic (`torch.randn_like`) where it needs a posterior mean.
- `RandomTimeShift` silently misaligns the anchor's t* unless the shift is recorded and applied.

Admission gate is additionally failing on a dirty tree. The 16 Aug target is not reachable.

---

## 4. Inventory — five publishable assets, ranked

### A. The azimuthal-harmonic account of acoustic metrics — STRONGEST, **but see the correction below**

> ## ⚠️ CORRECTION 2026-08-14 (same day) — the T60 keystone does not survive
>
> I recomputed both 27-checkpoint sweeps directly from the raw per-seed JSONs
> (`exp-09-cyl-dinov3-no-ssl/outputs_FLAC/{exp09_cylNoSSL,p1_sweep_import}`, K=8, 5 eval seeds/step).
> **Three claims in the original §4A are wrong.**
>
> **1. The cyl T60 win is a band-best spike, not a result.** cyl@40k T60 = **8.314 is the global
> minimum of all 27 cylindrical checkpoints**. Its immediate neighbours are **9.502 (37.5k)** and
> **10.293 (42.5k)** — over a full T60 point worse on *both* sides. P1's late band is flat by
> comparison (8.50 / 8.48 / 8.51). Aggregating over all checkpoints ≥ 37,500:
>
> | estimator | | T60 ↓ | C50 ↓ | EDT ↓ |
> |---|---|---|---|---|
> | **@40k (matched step)** | cyl mean-pool | **8.314** | 1.0800 | **39.09** |
> | | P1 vanilla | 8.904 | **1.0100** | 40.78 |
> | **band mean ≥37.5k** (13 vs 12 pts) | cyl mean-pool | 9.749 | 1.0423 | 43.20 |
> | | P1 vanilla | **8.761** | **0.9985** | **38.80** |
>
> Under band aggregation the cylindrical arm **loses all three metrics**. The quoted "8.24σ" is σ
> over *eval* seeds (±0.018); the relevant noise is checkpoint-to-checkpoint wander of ±1.0 T60,
> about 60× larger. This is exactly the failure mode logged as open issue #1 — and it caught me.
>
> **2. The original table compared different steps.** cyl@40,000 vs P1@55,000. At matched step 40k
> the **EDT claim reverses** — the cylindrical arm *wins* EDT by 1.7 ms (39.09 vs 40.78).
>
> **3. The strata are mismatched.** Verified from the JSON headers: every cyl cell is
> `cond_method=fa_invariant, cond_autocast=bf16`; every P1 cell is `vanilla, default`. Announcement
> 05 forbids exactly this mixing.
>
> **What actually survives every estimator: C50.** The cylindrical arm is worse on C50 at matched
> step (1.0800 vs 1.0100) *and* under band aggregation (1.0423 vs 0.9985). That is the m ≥ 1
> signature, and it is the only empirical leg of the harmonic account still standing.
>
> **A better, recipe-matched demonstration already exists.** Within the *same* arm family and recipe,
> max-pool vs mean-pool (neuronic exp_06 vs exp_05, @40k K=8 online): C50 **1.0592 vs 1.0811**
> (max better), T60 8.896 vs 8.622 (max worse); the same direction repeats in the MLP-head pair
> (exp_03 1.1112 vs exp_04 1.1518). Escaping m=0 buys C50 and costs T60 — demonstrated without any
> cross-recipe or cross-stratum comparison. **Max-pool is also exactly roll-invariant, so the paper's
> competitor is max-pool, not mean-pool.**
>
> **4. A factual error in the framing.** Vanilla DINOv3's `pooler_output` is the **CLS token**, not
> the token mean (`transformers/models/dinov3_vit/modeling_dinov3_vit.py`: `pooled_output =
> sequence_output[:, 0, :]`). "The standard readout is the m=0 projection" is false for the vanilla
> control. It is true of the *cylindrical port*, which replaced CLS with mean-pooling precisely to
> obtain invariance. Fix this sentence before it reaches a reviewer.
>
> **Consequence for the contribution.** Do not claim "beats mean-pool". Claim instead: mean-pool and
> max-pool are both **lossy** invariants; source-steered Fourier pooling is the **complete invariant**
> on the roll orbit — a bijection up to the group at full harmonic order. That claim is
> theorem-shaped, is unaffected by every correction above, and is what the experiment should test.
>
> **Design constraint discovered.** Under `fa_invariant` the source azimuth is *deleted* from the pose
> branch: `out["source"] = torch.stack([r_s, sz, torch.zeros_like(r_s)])`
> (`src/data/yaw_rotation.py`, and line 128 in both sibling checkouts). Every cylindrical arm trains
> this way; P1 does not. Source-steered pooling must therefore be shown to restore *source-relative
> directional structure*, not merely to re-inject an azimuth a config key removed. The magnitude-only
> control (|F̂[m]| without steering), which exp_11 already showed fails, is the ablation that separates these.


**Theory.** T60 is an **m=0 functional** of the room — Sabine's `0.161·V/(S·ᾱ)` is rotationally averaged — so an invariant, rotation-averaged readout is not merely adequate but variance-reducing. **C50 and EDT are not m=0**: they depend on source-relative early-reflection geometry, living in m ≥ 1. Mean-pooling *is* the projection onto m=0, so it provably cannot express them.

**Prediction:** an equivariant backbone with mean-pool readout wins T60 and loses C50/EDT.

**Confirmation, in data collected before the theory was written, across two independent architecture families.** Cylindrical DINOv3 vs its vanilla control at matched checkpoints (cyl@40k vs P1@55k), K=8:

| Cell | cyl@40,000 | P1@55,000 | Verdict |
|---|---|---|---|
| K1 T60 | 9.847 ± 0.052 | 9.808 ± 0.037 | parity (0.62σ_c) |
| **K8 T60** | **8.313 ± 0.021** | 8.488 ± 0.003 | **cyl BETTER (8.24σ_c)** |
| K1 C50 | 1.144 ± 0.002 | 1.030 ± 0.006 | P1 better |
| K8 C50 | 1.079 ± 0.002 | 0.954 ± 0.003 | P1 better (38σ_c) |
| K1 EDT | 41.79 ± 0.38 | 39.86 ± 0.35 | P1 better |
| K8 EDT | 39.08 ± 0.12 | 37.05 ± 0.08 | P1 better |
| K8 R@1 | 5.48 ± 0.11 | 6.07 ± 0.08 | P1 better (4.3σ_c) |

Reproduced on fresh seeds 47–51 (all six cells).

**A failed intervention that the theory predicts should fail.** exp_10 distillation from a task-aligned teacher: pre-registered A∧B∧C = FAIL. C50 improved (+15.6σ_c / +23.1σ_c vs cyl@40k, closing 67.1%/57.2% of the gap) but **the T60 inversion was spent** (guards tripped at +2.66σ / +28.30σ). Reading: you cannot distil back information the readout already discarded. Equivariance held exactly throughout (9.455e-07).

**A negative probe consistent with the theory.** exp_11 readout probe: all five fixed-linear invariant readouts scored *negative* vs the plain mean (best R2 at −0.294%, worst R4+PCA1024 at −1.784%). Consistent with the missing structure being nonlinear and phase-bearing — the probe excluded learned DeepSets φ and the source-steered phase readout **by design**, so C2 is not refuted.

**The derived fix — never run.** **Source-steered Fourier pooling**, `F̂[m]·e^{imφ_s}`: exactly invariant *and* lossless, because it retains m ≥ 1 phase relative to the source. Roughly one MLP, no backbone change.

**Partial support already in hand.** exp_12 arm A (C3 low-band RoPE + C4 m0-restricted registers) beats the naive cylindrical arm on **6/6 metrics at both K**, closing at K=8: T60 67.8%, **C50 85.7%**, EDT 36.1% of the gap to vanilla FLAC. EDT remains the laggard.

⚠️ **Caveat that must be handled honestly:** the motivating variance decomposition (mean-pool retains 61.4% of between-scene variance; 88% of the discarded 38.6% lives in m=1..4; modes m0 68.2% / m1 19.6% / m2 3.9% / m3 3.0% / m4 1.4%) is flagged **[E] exploratory** — 22 samples, one source index, a pretrained rather than task-trained backbone, unreviewed. It motivates the theory; it cannot be the evidence for it. Either re-run properly at scale or present explicitly as motivation.

### B. Group order vs pose robustness — the C4 trap — READY NOW

exp_14 yaw_gen, closed 12 Aug. 106/106 cells valid at one pin, gates G1–G4 all pass, G5 external reproduction exact (ΔT60 = 0.0000 vs exp_11's committed rows). Degradation under uniform random yaw, K=8, paired:

| Orbit | ΔT60 (random yaw) | Reading |
|---|---|---|
| VANL (none) | +0.521 ± 0.037 | degrades |
| C4 | **+0.531 ± 0.029** | **buys nothing** (p ≈ 0.5 vs VANL) |
| C8 | +0.049 ± 0.011 | ~10× flatter (p = 4.3e-6) |
| C16 | −0.003 ± 0.018 | fully invariant |
| C32 | +0.006 ± 0.011 | fully invariant |

A sharp dose-response with saturation between C8 and C16. And the finer orbits that *do* confer robustness cost accuracy at canonical pose (C16 T60 9.343 vs C4's 8.414) — a clean accuracy–robustness trade curve.

Pre-registered verdicts: H-P PARTIAL (C32 wins R@1 +0.650, Holm p = 0.0034; loses T60 +0.250, p = 9.2e-5); H-M SUPPORTED; H-S SUPPORTED.

Descriptive: **C8 is the efficient point** — under random yaw it ties VANL on T60 (7.726 vs 7.724), beats it on C50 by −9% (0.868 vs 0.954) and R@1 by +17% (5.20 vs 4.44), costing EDT +1.75.

Generalises well beyond acoustics — directly actionable for anyone applying frame averaging or canonicalization to panoramic input.

### C. How symmetry gains get overstated — a measurement contribution — READY NOW

- **Checkpoints are band draws, not trajectory points.** Non-decaying LR → adjacent checkpoints swing ±0.5 T60, ~1.5× the effect size being measured. Three of our own conclusions were distorted before this was quantified. exp_13 showed a decaying tail halves the band but converges to a *different* metric trade point rather than reproducing a wide-band best draw.
- **Protocol mismatch is catastrophic and silent.** The same FA checkpoint reads `8.202 / 0.978 / 38.79 / R@1 5.39` under matched conditioning and `10.652 / 2.082 / 80.86 / R@1 0.68` under the default. This caused exp_09's protocol error and one retracted exp_07 conclusion.
- **Symmetry benefit is a train×eval interaction.** The 2×2 at 40k, K=8 shows inference-only frame averaging on vanilla weights actively hurts (R@1 5.192 → 4.049, EDT +1.66). Only FA-trained + FA-eval is good.

### D. Cylindrical DINOv3 and the XYZ gauge — ARTIFACT

Exact C₃₂ azimuth equivariance on the **official pretrained** DINOv3 ViT-S/16 with complete weight inheritance (`missing_keys == [] and unexpected_keys == []`): patch error **2.2590e-06**, pooled **1.2683e-06**, vanilla control **4.9405e-01**. Reproduced to 10 s.f. by an independent reviewer.

The XYZ gauge is the more interesting half and makes a good figure: array-roll equivariance is **not** physical-yaw equivariance when channels are 3-vectors — gauge ON 5.2125e-06 vs OFF 2.4586e-01, a **4.7×10⁴×** separation.

⚠️ Position carefully against **ViewRope (arXiv 2602.07854)**, contemporaneous prior art on low-band geometry RoPE. Our claim is the verified port with strict checkpoint compatibility, not the mechanism.

### E. ARE — analytic direct-path anchor — BLOCKED

Reparameterise the rectified-flow target as a residual against an analytic direct-path anchor, aimed squarely at EDT — the one metric that has resisted every arm. The anchor is analytically yaw-invariant so it would compose with the equivariance line rather than compete. Genuinely promising as a *next* paper; not for this one (see §3.1).

---

## 5. Recommended narrative

**Recommended title** (5-framing generation → 3-lens judge panel, unanimous top-1, mean 9.17/10, no vetoes):

> **Reverberation Time Is Rotation-Averaged, Clarity Is Not: An Azimuthal-Harmonic Account of Invariant Readouts in Room Impulse Response Generation**

Why this construction survives everything that could go wrong:

- The main clause is a **property of the acoustic functionals themselves** (Sabine is a pure global average; C50/EDT depend on source-relative early reflections). No experimental outcome, single-seed objection, or recipe reversal can falsify it.
- **"Invariant readouts", not "equivariance"** — attribution sits on the mean-pool projection onto m=0, not on the symmetry. This matters: if source-steered pooling later works, an *equivariant-and-accurate* model would be a counterexample to a title that blamed equivariance.
- **"An … Account of"** pre-registers the understanding-paper register, with no method obligation.
- "Room Impulse Response Generation" (not "Room Acoustics") carries the high-value query string.

Two usage notes: put **"yaw-equivariance" in the abstract's first sentence and the keyword field**, since the title now carries only "invariant"; and gloss **clarity (C50)** and **early decay time (EDT)** in the first two lines, so no ML reviewer reads "clarity" as a vague English word.

**Conditional variants**

| If | Title |
|---|---|
| Seeds 43/44 soften the metric split (K=8 T60 inversion narrows toward parity) | *What Rotation-Invariant Readouts Can and Cannot Encode: Azimuthal Harmonics in Room Impulse Response Generation* — stakes the claim on provable expressivity rather than a measured split, immune to the n=1 exposure |
| The source-steered arm does not run or clearly fails | *Exact Yaw Invariance Does Not Improve Room Impulse Response Prediction, and Azimuthal Harmonics Explain Why* — only with the seed replication **and** the exp_15 augmentation row in hand |
| Source-steered pooling lands | Keep the main clause, swap the subtitle: *…: Source-Steered Fourier Pooling for Room Impulse Response Generation* |
| It lands large enough to be the centerpiece | Go method-forward: *Source-Steered Fourier Pooling: Exactly Yaw-Invariant Readouts That Preserve Azimuthal Structure in Room Impulse Response Generation* |

**Patterns eliminated**, beyond the usual banned list: leaderboard formulations ("Equivariance Wins T60 and Loses C50") — the directional claim is n=1 and reverses between recipes, and it credits equivariance rather than the readout; slogan-only main clauses ("Invariance in the Wrong Place") — they promise a demonstrated *right* place and leave the subtitle as a bare topic tag; any title blaming "equivariant models", for the counterexample reason above; and question forms, which read softer at ICLR.

⚠️ The title follows from decision 1 in §8. If frame averaging stays the headline rather than the foil, every title above is wrong.

---

Frame averaging appears as the **setup and the foil**, never as the contribution.

1. **Physics says the RIR field is yaw-equivariant. The models are not.** Rotating the panorama by an angle that changes nothing physical moves the prediction by 0.193–0.221 relative L2 and degrades every metric by 9–23σ, worst at 180°.
2. **Both obvious fixes deliver machine-precision invariance.** Frame averaging reaches 4.9e-08 at the conditioning level; the equivariant backbone reaches 2.3e-06. The symmetry itself is not in doubt anywhere in the program.
3. **And neither improves accuracy at canonical pose.** Present the recipe-contingency as a controlled result rather than a caveat — this is the honest core, and it is interesting precisely because the equivariance literature reports positive results almost uniformly.
4. **Why: the symmetry was imposed in the wrong place.** Invariance was applied to the *representation*, by projecting onto m=0, rather than to the *prediction*. T60 survives because it is m=0; C50 and EDT are destroyed because they are not.
5. **The fix follows from the diagnosis.** Source-steered invariant pooling, low-band RoPE, m0-restricted registers — exactly invariant without discarding m ≥ 1.
6. **What invariance does buy is pose robustness** — the dose-response curve and the C4 trap.
7. **Practical guidance** — evaluation protocol, the band-draw warning, group-order choice.

If beat 5 lands, this is a positive-result paper. If it only partially lands, beats 1–4 and 6 still make a solid ICLR "empirical understanding" submission — a well-established genre there. Either way the paper does not rest on a single contested delta.

### Novelty check

Equivariance in RIR generation is essentially unexplored — a literature sweep over neural acoustic fields (NeRAF, NACF, AVR, RAF benchmark) turns up no symmetry treatment. Negative-and-mechanistic results on imposed equivariance *do* exist (2D-to-3D pose lifting reports the same accuracy-for-robustness trade; canonicalization work reports trajectory-crossing / conflicting-gradient pathologies for symmetric flow models), which helps rather than hurts: the framing is legible to reviewers, and we extend a recognised line into a domain where the harmonic structure gives it real explanatory teeth.

⚠️ Missing baseline reviewers will ask for: **canonicalization** (map to a canonical pose, train unconstrained, sample a random transform at generation). Worth a paragraph even if not run.

---

## 6. Five-week plan (abstract 18 Sept, paper 25 Sept)

Training must stop ~10 Sept to leave room for 5-seed evaluation and writing. ~4 usable weeks.

### Cut first

| Cut | Frees | Rationale |
|---|---|---|
| **DS-CS3** (drawshare arm 2) | ~8 GPU-days | Resolves *why* the FA comparison reversed. Under the recommended narrative the reversal is a reported finding, not the headline; reporting both recipes honestly suffices. Let DS-PA finish for the per-angle replication, then stop. Cheaper substitute exists if needed: exp_11's analysis names a screen-level A/B on C8, where the chunk-sharing delta is largest. |
| **exp_11 100k legs** | the cluster | 4 of 5 never started; C32 alone is ~160 h and changes no conclusion. The orbit question is answered at 40k. |
| **ARE** (defer) | A6000 slot | Fix the four blockers properly, run after the deadline — or only if a pool frees unexpectedly. |

### Then run, in this order

1. **Source-steered readout arm** — design this week, run on the cluster (~13 h L40). The only experiment that converts this from a negative to a positive result. Needs plan + one review round under SOP (~1 day). Nothing else starts before it is queued.
2. **Two more training seeds (43, 44) on the headline pair** — the single highest reviewer-risk reduction. Every arm in both repos is n=1, and claimed effects are smaller than the spread between our own controls.
3. **Protect exp_15** (random-yaw augmentation baseline, leg 1/16 launched 14 Aug) — every reviewer asks "why not just augment?"; the paper is incomplete without this row under any narrative.
4. **exp_12 arm C** — finishing anyway, free second point on the RoPE-design axis.

---

## 7. Reviewer attack surface

- **Single training seed.** The one that actually sinks papers. Addressed by plan item 2.
- **"Your best model has no symmetry."** Own it in the abstract — it is the paper's point.
- **One dataset (AR only).** A small HAA transfer table would disproportionately help; the finetune recipe exists in-repo (`--max-steps 1000`).
- **The m=1..4 variance decomposition is exploratory** (22 samples, one source index, pretrained backbone, unreviewed). Re-run at scale or present explicitly as motivation only.
- **ViewRope prior art** on low-band geometry RoPE needs a careful positioning paragraph.
- **Missing canonicalization baseline.**
- **Scope discipline.** The corpus is enormous. Most of exp_03–06, 08, 09, 13 will not appear; trying to include them produces an unreadable paper.

---

## 8. Decisions awaiting Yixun

1. **Which paper?** *Recommended:* the harmonic account, FA as foil.
2. **May DS-CS3 and the 100k legs be cut?** *Recommended:* yes — frees ~8 GPU-days and unblocks the cluster. This is the decision that makes everything else fit. Not done unilaterally because it stops work already in flight.
3. **Design the source-steered readout arm now?** *Recommended:* yes, starting today.
4. **ARE — defer or promote?** *Recommended:* fix blockers, defer past the deadline.
5. **The DS-PA pause tonight (~20:00)** recorded in exp_16's worklog is another session's *interpretation* of intent, not a confirmed instruction. Needs an explicit yes/no.
6. **Authorship** — ICLR 2027 has new co-authorship and reciprocal-reviewing policies; zhixuan's `Yaw-equi-ViT` is merged into this repo.

---

## Provenance

Numbers are transcribed from the experiment records, not recomputed here. Primary sources:

- `FLAC/worklog/worklog_yixun/model_comparison.md` (living table, all 5-seed rows)
- `FLAC/worklog/worklog_yixun/exp_02_yaw_noninvariance_claude/`, `exp_07_fa_scratch_claude/` (+ `fa_scratch_CORRECTION_addendum.md`), `exp_10_fa_scratch_resume_claude/`, `exp_11_fa_orbit_claude/fa_orbit_{results,analysis}.md`, `exp_13_decay_tail_claude/`, `exp_14_yaw_gen_claude/yaw_gen_results.md`, `exp_14_fa_drawshare_claude/plan_fa_drawshare.md`, `exp_15_yaw_aug_claude/`, `exp_16_are_port_claude/{plan_are_port.md,are_port_codex_plan_review.md}`
- `cylindrical-dinov3/worklog/worklog_yixun/PHYSICS_CONSTRAINTS_synthesis_exp05_to_exp10.md` ← the key synthesis
- `cylindrical-dinov3/worklog/worklog_yixun/{exp_01,exp_03,exp_06,exp_10,exp_11,exp_12}`, `worklog_yixun_neuronic/model_comparison_neuronic.md`

Rendered copy: `ICLR2027_narrative_review.html` (same folder, open in a browser).
