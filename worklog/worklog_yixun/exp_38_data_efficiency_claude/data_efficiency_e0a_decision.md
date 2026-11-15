# exp_38 — E0a decision (signed off by Yixun, 2026-11-08; widened the same day)

**Machine-readable, sealed record:** `data_efficiency_e0a_decision.json` (+ `.sha256`, schema 2,
decision `E0a-retrieval-R@1-5-10-2026-11-08`). The E0 gate (`exp38_eval.py e0` / `validate_e0_receipt`) consumes
ONLY that file; this page is its human-readable copy. It supersedes the R@1-only record
`E0a-retrieval-R@1-2026-11-08` (sha256 `332338a3…`, commit `1a158d73`), whose four R@1 pins it keeps unchanged.

- **Gate:** E0a — the replay of exp_14's `dc_van_f025@40k` (`aa1a2cd6…`, `FLAC_AR_BVp1.json`) at the exp_38
  evaluator pin, K8 s42 and K1 s42, against exp_14's own records (`ref_cells/E0/E0a_K{8,1}_s42.json`).
- **Observed (E0 at `bfc4ea4b`, 2026-11-08 01:57–03:44, K8 on card 0, K1 on card 1):** T60, Invalid T60, C50, EDT
  and FD are type-strictly identical. The AGREE retrieval metrics differ by 1–6 items of 6,337: 5 of the 11
  `metrics` keys per K, all retrieval. E0b–E0d are exact. The coordinator independently compared the outputs with
  `ref_cells` and confirmed these values. Copies of the two metrics files (sha256 K8 `9b1c5378…`, K1 `e3c148f6…`,
  equal to the bfc4ea4b E0 receipt's) are in `src/tests/fixtures/exp38_e0a_replay_bfc4ea4b/`.
- **Questions (AskUserQuestion, main session, relayed by the coordinator):**
  1. Accept the exact acoustics, or stop? Answer: **"Accept, label R@1"**.
  2. The same replay also differs at R@5 / R@10: widen the accepted scope, or stop? Answer: **"Widen to R@1/5/10"**
     — the E0a accepted deviations cover all retrieval metrics.
- **Reason (recorded in the decision):** AGREE's audio encoder samples noise at evaluation. `VAEBottleneck.encode`
  calls `vae_sample` (`AGREE/AGREE/audio_model.py`: `latents = torch.randn_like(mean) * stdev + mean`, no eval
  guard), so the retrieval embeddings, and hence R@k, depend on the RNG state at the moment of evaluation. exp_37's
  G0 diagnosis found the same cause the same day.
- **Scope (narrow; the ten measured values pinned):**

| cell | key | ours (exp_38 evaluator) | exp_14 record | items (of 6,337) |
|---|---|---|---|---|
| de_E0a_K8_s42 | RIR_to_GT_RIR_R@1 | 4.4343 | 4.3869 | +3 |
| de_E0a_K8_s42 | RIR_to_GT_RIR_R@10 | 22.8657 | 22.8184 | +3 |
| de_E0a_K8_s42 | RIR_to_geom_R@1 | 3.519 | 3.5506 | −2 |
| de_E0a_K8_s42 | RIR_to_geom_R@5 | 12.6243 | 12.719 | −6 |
| de_E0a_K8_s42 | RIR_to_geom_R@10 | 19.4572 | 19.4729 | −1 |
| de_E0a_K1_s42 | RIR_to_GT_RIR_R@1 | 4.4343 | 4.4185 | +1 |
| de_E0a_K1_s42 | RIR_to_GT_RIR_R@5 | 14.4864 | 14.5337 | −3 |
| de_E0a_K1_s42 | RIR_to_geom_R@1 | 3.5348 | 3.5032 | +2 |
| de_E0a_K1_s42 | RIR_to_geom_R@5 | 12.2455 | 12.1982 | +3 |
| de_E0a_K1_s42 | RIR_to_geom_R@10 | 19.4887 | 19.394 | +6 |

- **Everything else stays type-strictly exact:**
  - all acoustic metrics and FD of E0a;
  - E0a's undecided retrieval keys (K8 GT R@5, K1 GT R@10);
  - every key of E0b–E0d.

  A re-run E0a passes only when its deviations are a subset of these keys **with exactly these values**. A different
  random draw in the re-run fails closed, and the E0 then stops. The loader refuses any decision outside this shape:
  another gate or cell, a non-retrieval key, a reference that is not the registered record, ours == reference, a
  missing reason, or an empty scope.
- **Downstream:** the FLAC 25/50/75 % rows (exp_14's dc_van records) keep exp_14's values.
  - In the table, every shown retrieval column of these rows carries `†`, and the footnote names R@1, R@5 and R@10
    for both RIR→GT-RIR and RIR→geom.
  - In the results JSON these rows carry `retrieval_evaluator: "exp_14"` and `exp_14_evaluator_metrics` = all six
    R@k keys (`exp38_collect.py`).
