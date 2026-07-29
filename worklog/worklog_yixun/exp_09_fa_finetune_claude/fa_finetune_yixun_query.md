# Queries — exp_09 fa_finetune

## Q1 (2026-07-29, commissioning)
**Verbatim:** "I agree with that we should target for exp_09 — the fa equivariant fine-tune from the new 87.5k anchor, go ahead"
**Summary:** Commission exp_09: fine-tune the fa_invariant (C₄ frame-averaged) conditioning from the exp_07 checkpoint of record (`exp07_P1/.../epoch=19-step=87500.ckpt`, full-parity vanilla anchor).
**Assumption/hypothesis:** exp_08 proved fa fine-tuning from a converged vanilla model yields exact C₄ invariance with seed-robust T60 gains; exp_07 proved fa-from-scratch fails and delivered a superior-or-equivalent anchor WITH full optimizer/EMA state (unlike the released EMA-only ckpt — the original Route-B motivation). Fine-tuning from the stronger, state-complete anchor should give the equivariance win without the fine-tune-damage seen when starting from the released ckpt.
**Why run:** completes the project narrative — "yaw-equivariant FLAC that beats the released Table-1" — by combining exp_07's anchor with exp_08's route.
