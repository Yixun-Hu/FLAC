# Lab notebook — exp_09_fa_finetune

## 2026-07-29 — scaffold
- **Goal** — fa_invariant (C₄ frame-average) fine-tune from the exp_07 full-parity anchor (`exp07_P1/.../epoch=19-step=87500.ckpt`; vanilla, SyncBN-64 DDP recipe, full optimizer+EMA state on disk).
- **Version Control** — branch check-equivariance-necessity, base = exp_07 closure (`a19499a`).
- **Resources at scaffold** — both GPUs free; env `flac`; wandb yh4742.
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → TDD (if any code) → runs.

## 2026-07-29T13:47:43-04:00 — plan Rev 2 APPROVED by Yixun ("Approve — proceed") → implementation round 1
- **Goal** — code round: `f_arm_launch.sh` (MODEL_CONFIG-parameterized p1_ddp_launch mirror) + opt-reset hook (strips optimizer state from a ckpt COPY; anchor never mutated). Coder = Opus 5 max (SOP 2026-07-25); TDD for Python; Codex review before use.
- **Next** — code → review → resume-validation probes (both arms) → 1,250-step F-warm/F-reset probes → fixed-rule pick → committed runs.

## 2026-07-29T20:15:05-04:00 — variant probes DONE → PICK = F-warm (fixed rule, unambiguous); committed runs launching
- 15-step resume probes: both PASS (Fw full-state restore; Fr stripped-copy verified: adam CLEARED, param_groups KEPT @4.794633e-5, anchor SHA intact).
- 1,250-step probes (online @88750): Fw EDT 40.352/R@1 5.981 vs Fr 40.708/5.918 → **Fw wins both** → F-warm continues (per plan: resume its own 88750 ckpt) to 97.5k. EMA @88750: Fw 9.000/1.0394/37.562/6.170. Both variants show the fa-adaptation transient (R@1 dip at +625, recovery by +1250) — expected.
- Next: F-warm 88750→97500 (wandb), then V control (anchor→97500, BVp1). Screens 2.5k-cadence after each; then G1–G4.

## 2026-07-30T12:31:49-04:00 — F-warm committed run DONE (97.5k) + screens: fine-tune-damage signature — G2 heading to FAIL (pre-registered NEGATIVE branch)
- Fw screens (EMA s42): 90k 8.785/1.0232/38.307/R5.713 · 92.5k 8.990/1.0426/41.511/R5.207 · 95k 8.686/1.0466/38.883/R5.129 · 97.5k 8.921/1.0626/44.030/R4.702 — ALL worse than the anchor (8.293/0.966/35.95/6.96) on ALL metrics, degrading monotonically; probe window (88.1–88.7k) was the peak. C50 alone ~19σ outside the candidate band → **no G2 qualifier possible** on this curve.
- Read: exp_03–06's damage law reproduces from the warm-state SyncBN-64 anchor at native-schedule lr (4.79e-5). Pending before the verdict: V control (G4 ΔΔ — drift vs fa-specific) + G1 equivariance block on the best F ckpt (88750). If G1 passes, the finding = equivariance achieved at quality cost at this lr → the pre-registered reduced-lr option goes to Yixun.

## 2026-07-30T19:02:05-04:00 — ⚠️ PROTOCOL ERROR FOUND: all Fw evals ran with cond_method='vanilla' (eval_FLAC --cond-method never passed) — damage curve is a train/eval-mismatch candidate; G1 non-flatness EXPECTED under vanilla eval
- Evidence: rot-sweep JSON records cond_method='vanilla', frame_avg_angles=None; eval_FLAC imports invariant_conditioning + has a --cond-method flag (filename suffix `_fa_invariant_a4`) we never used. Scope: ALL exp_09 Fw evals AND exp_07 B-F screens (its from-scratch conclusion needs a robustness re-check under fa eval; magnitude 2× likely survives, but must be verified).
- CORRECTIVE BLOCK (launched): Fw-88750 with --cond-method fa_invariant — rotation sweep 0/90/180/270 + 45 control, and the fine-tune curve 90k–97.5k. G1/G2 adjudication re-runs on these.

## 2026-07-30T22:16:53-04:00 — VERDICT: G1 PASS (exact C₄: rotations identical to 3–4 decimals; 45° control breaks) + Fw-95000 fa-eval 5-seed both K: 4 SUPERIOR + 1 EQUIV + 2 NONINF + 1 OUT (K8 EDT +0.40) → **PARTIAL tier, at released-Table-1 level**
- K8: T60 8.4652±0.0058 SUP(−10.8σ) · C50 0.9582±0.0010 SUP(−3.2σ) · EDT 37.4968±0.0813 OUT(+3.7σ) · R@1 6.9243±0.0701 NONINF(−1.1σ). K1: 9.8271±0.0612 SUP · 1.0337±0.0025 SUP · 40.8740±0.3393 NONINF · 6.8581±0.1108 EQUIV(+0.11σ, above released).
- G2 vs anchor: C50 better, R@1 ≈ equal, T60/EDT band-scale cost within the V control's drift band (G4 ΔΔ ≈ 0). Yesterday's "damage" = the vanilla-eval protocol artifact, retired.
- **Checkpoint of record (equivariant): `outputs_FLAC/exp09_Fw/.../epoch=20-step=95000.ckpt`, eval with `--cond-method fa_invariant`.** Closing package (incl. protocol-error record + exp_07 B-F fa-eval robustness note) = tomorrow.
