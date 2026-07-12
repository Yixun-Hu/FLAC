# Lab notebook — exp_05_bn_drift_bisect

## 2026-07-06T01:21:30-04:00 — scaffold
- **Goal** — BN-drift measurement + loader-knob bisection + gated validation, per exp_04 analysis; commissioned by Yixun.
- **Version Control** — branch check-equivariance-necessity, base_commit 6220b67 (end of exp_04).
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → TDD.

## 2026-07-06T01:24:49-04:00 — approval delegation (recorded)
- **Goal** — capture Yixun's standing approval for autonomous overnight execution.
- **Result** — Yixun (verbatim): "If review and modification and then codex approve, then I will automatically approve. I will go to bed. Please run automatically." ⇒ SOP user-approval step for THIS plan is satisfied automatically once the plan review's findings are addressed and no REQUEST-CHANGES remains outstanding; all subsequent gates (per-round code reviews, V1/V2 stop rules, marginal-pass pauses) remain in force — a marginal V2 pass HOLDS for Yixun per the registered rule rather than auto-proceeding.
- **Next** — plan review verdict → revise → auto-approve → TDD cycle.

## 2026-07-06T02:04:52-04:00 — round bndrift CLOSED; B-1/B0 launching
- **Version Control** — write `3f7635a`+`fd2daa7` → review `4a15da7` (REQUEST-CHANGES ×6; Welford/hook-target/B×N/test-constant all CONFIRMED correct) → fix `ee31582` (+155/−23; CUDA smoke ran and passed; all-buffer no-mutation guard proven by train-mode negative test) → Planner re-verified: 97 passed.
- **Result** — `passed`; round CLOSED. Instrument hardened: fail-fast load (exp_03 confound class impossible), device-correct accumulators, output-hook bug class killed (>1.0 divergence vs 5.66e-6 identity gap), buffer bit-identity asserted inside the probe itself.
- **Acceptance (B-1, pre-registered):** clean load 0/0, 20 BN layers, no-mutation assertion passes, worst-layer table produced from ONE real batch.
- **Acceptance (B0, pre-registered):** 3 repeats × 200 batches × batch 16 (train loader) + 1 × eval loader; report per-layer mean-shift/var-ratio with cross-repeat min/max; drift structure interpretable (which layers/depths carry it).
- **Next** — B-1 pilot → B0 → grid.

## 2026-07-06T02:11:33-04:00 — B0/B1-stage-1 verdict: current loader is the optimum; EMA-tail hypothesis; AMENDMENT drafted
- **Result (B0, 3 repeats, tight ±0.001–0.02):** all 20 layers over threshold on the train loader; monotone depth amplification (stem 0.082 → layer4 0.357); eval-loader reference 2–4× worse (validates probe discrimination). Load 0/0, no-mutation held.
- **Result (B1 stage 1, max_len grid):** 4800 → 0.65 stem / 1.12 max; 10240 → 0.12 stem / 1.39 max; 19200 → 0.62 / 1.68. **Baseline 9600 is the clear minimum — reviewer's suspect #1 exonerated; metadata-time truncation config as shipped matches the original.**
- **Hypothesis (registered before any further action):** the residual baseline drift is EMA-tail estimation noise of the released running stats (momentum 0.1 ⇒ stats ≈ EMA over the last ~tens of training batches; expected per-channel EMA sampling noise ≈ batch-mean std × √(m/(2−m)) ≈ same 0.05–0.1σ order as measured stem drift), amplified with depth — i.e., NOT a loader knob. Prediction: stage-2 knobs (normalization/padding/onset) would behave like stage 1 (all worse than baseline).
- **Amendment (sent for Codex review per delegation):** replace stage-2 knob grid + V1-under-corrected-loader with **V1′ = BN-frozen fine-tune validation**: add `--freeze-bn` to finetune_cond.py (RIR-encoder BN modules to eval() during training + running stats pinned; affine params still trainable... exact scope per review), rerun the R1b-recipe vanilla control with BN frozen → exp_01 gate. Rationale: neutralizes the W0-proven damage channel regardless of drift provenance; W0 (~30% damage, pure BN) becomes a no-op by construction; tests whether the ~70% gradient component was co-adaptation to shifting BN stats.
- **Next** — Codex amendment review → auto-approve if clean → TDD (freeze-bn) → V1′.

## 2026-07-06T02:18:32-04:00 — dispersion check: EMA-tail REFUTED as sole cause; V1′ proceeds (provenance-agnostic)
- **Result** — stem-BN per-batch dispersion over 60 batches: predicted EMA-tail noise max 0.024 vs observed shift max 0.085 (≈3.5×; B0's 0.082 reproduced) → residual is genuine pipeline/content drift, small at stem, no max_len knob fixes it (stage 1). Registered hint applied honestly: "pipeline-drift if observed >> predicted". Side discovery logged: train loader applies augmentations (Random Time Shift, Add Noise) to the main audio; context path loads raw — augmentation lineage now an additional provenance unknown, further motivating the provenance-agnostic repair.
- **Decision** — V1′ (BN-freeze validation) proceeds exactly as amendment-approved: its rationale never depended on drift provenance; it eliminates the W0 damage channel by construction and tests the co-adaptation share of the gradient component.
- **Next** — freeze-bn TDD round (7 review-mandated tests) → per-round review → smoke → V1′ launch.

## 2026-07-06T02:33:03-04:00 — round freezebn CLOSED; V1′ smoke + launch
- **Version Control** — write `d1c4e5c`+`5d1c64c` (104 tests, Planner re-verified) → review APPROVE-WITH-NITS (enforcement verified against installed PL 2.1 source: no one-batch BN-mutation gap on any path; scope = exactly the RIR encoder's 20 BNs) → CLOSED, no fix leg.
- **Acceptance (smoke):** '[finetune_cond] FreezeBN: 20 BatchNorm modules frozen' logged; 10 steps finite loss.
- **Acceptance (V1′, pre-registered):** R1b recipe + --freeze-bn (NO warmup — one variable); exp_01 2σ gate, marginal band 1.5–2σ pauses; PASS ⇒ resume pipeline as fa_invariant + freeze-bn; FAIL ⇒ registered stop (gradient-path damage independent of BN) + analysis.

## 2026-07-06T06:27:37-04:00 — V1′ verdict: FAIL → REGISTERED STOP; per-metric decomposition complete
- **Result (V1′, freeze-bn, 5 seeds, full split):** K=1 T60 10.523±0.058 (7.9σ), C50 1.010±0.007 (3.7σ **in the improved direction — better than baseline**), EDT 41.33±0.12 (3.5σ); K=8 T60 9.235 (48.4σ), C50 0.928 (10.4σ, improved), EDT 38.73 (24.2σ). R@1 at baseline (0.24σ/0.67σ).
- **Decomposition (vs R1b unfrozen / W0 lr=0 / baseline):**
  - **EDT:** largely BN-mediated — freeze-bn recovered K=1 43.27→41.33 (residual +1.38 vs baseline ≈ gradient part; W0's pure-BN was +1.15).
  - **C50:** BN-mediated and then some — freeze-bn overshoots to BETTER-than-baseline at both K (original running stats + trainable affine = improved clarity).
  - **T60:** gradient-driven, BN-independent — 10.47→10.52 (K=1), 9.20→9.23 (K=8): freezing changed nothing.
- **Analysis** — every optimizer-side hypothesis is now falsified (Adam transient, EMA, batch noise beyond partial, lr, BN mutation). The T60-specific residual is a genuine training-lineage difference on the gradient path (objective/data lineage), unreachable by recipe repair from the released artifact. Registered stop honored: no fa_invariant fine-tune as an absolute-accuracy (H3 vs exp_01) claim.
- **Next** — exp_05 closes with results/analysis; decision package for Yixun: (A) **matched-comparison route** [now well-founded: fa_invariant+freeze-bn vs vanilla+freeze-bn at identical recipe — both arms carry identical known damage; FA's marginal effect + H1/H2 rotation sweeps on the FT model become cleanly measurable], (B) from-scratch fa_invariant training [gold standard for Table-1 goals], (C) further lineage bisection on the gradient path [open-ended].
