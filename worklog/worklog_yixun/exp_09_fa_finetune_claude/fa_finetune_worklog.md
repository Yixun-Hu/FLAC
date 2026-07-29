# Lab notebook — exp_09_fa_finetune

## 2026-07-29 — scaffold
- **Goal** — fa_invariant (C₄ frame-average) fine-tune from the exp_07 full-parity anchor (`exp07_P1/.../epoch=19-step=87500.ckpt`; vanilla, SyncBN-64 DDP recipe, full optimizer+EMA state on disk).
- **Version Control** — branch check-equivariance-necessity, base = exp_07 closure (`a19499a`).
- **Resources at scaffold** — both GPUs free; env `flac`; wandb yh4742.
- **Result** — `launched` (planning). Plan → Codex plan review → Yixun approval → TDD (if any code) → runs.

## 2026-07-29T13:47:43-04:00 — plan Rev 2 APPROVED by Yixun ("Approve — proceed") → implementation round 1
- **Goal** — code round: `f_arm_launch.sh` (MODEL_CONFIG-parameterized p1_ddp_launch mirror) + opt-reset hook (strips optimizer state from a ckpt COPY; anchor never mutated). Coder = Opus 5 max (SOP 2026-07-25); TDD for Python; Codex review before use.
- **Next** — code → review → resume-validation probes (both arms) → 1,250-step F-warm/F-reset probes → fixed-rule pick → committed runs.
