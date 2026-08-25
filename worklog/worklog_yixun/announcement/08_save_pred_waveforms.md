# 08 — Predicted RIR waveforms are a required run artifact (standing, Yixun 2026-08-20)

## Original instruction (verbatim)
> If not, please be aware that not only AGREE similarity is saved, but also the RIR waveform (pred waveform) should be saved.

## Rule
- Every generative localization/eval run saves the predicted (generated, decoded, clamped — i.e. exactly-as-scored) RIR waveforms, not only their embeddings/similarities.
- exp_18 implementation: per-query `.npy` stacks `[M, K, 10240]` float32 (the exact scored tensors), stored under `/media/diskstation/yixunhu/FLAC/exp18_pred_waveforms/<cell stem>/`, per-query sha256 + relative path recorded in the row; dump-dir manifest published with the run.
- Runs completed before this rule (R2 seeds 42/43/44) get a regeneration-with-verification pass: the deterministic noise bank re-derives waveforms bit-exactly, and the pass must reproduce the published per-sample sims bitwise — doubling as an integrity audit.
- Applies to all future experiments (cross-arm exp_20+ included).

## exp_22 exemption (Yixun, 2026-08-24, verbatim decision "2e: approved")
Mesh-grid localization (exp_22) generates ~25M candidate-query pairs × K samples — full waveform dumps are physically impossible (~PB). Approved bounded rule: dump ONLY the 16 pre-registered off-grid probe queries plus the quantile-selected visualization cases; every similarity/score is logged as usual. This exemption is experiment-specific; the default rule stands elsewhere.
