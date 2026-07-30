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
