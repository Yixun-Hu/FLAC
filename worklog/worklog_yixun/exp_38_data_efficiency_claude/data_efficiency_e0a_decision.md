# exp_38 — E0a decision (signed off by Yixun, 2026-11-08)

**Machine-readable, sealed record:** `data_efficiency_e0a_decision.json` (+ `.sha256`, sha256 `332338a3…`). The E0
gate (`exp38_eval.py e0` / `validate_e0_receipt`) consumes ONLY that file; this page is its human-readable copy.

- **Gate:** E0a — the replay of exp_14's `dc_van_f025@40k` (`aa1a2cd6…`, `FLAC_AR_BVp1.json`) at the exp_38 evaluator pin, K8 s42 and K1 s42, against exp_14's own records (`ref_cells/E0/E0a_K{8,1}_s42.json`).
- **Observed (E0 at `bfc4ea4b`, 2026-11-08 01:57–03:44, cards 0/1):** T60, Invalid T60, C50, EDT and FD are type-strictly identical; the AGREE retrieval metrics differ by a few items. E0b–E0d are exact.
- **Question (AskUserQuestion, main session):** accept the exact acoustics, or stop?
- **Answer (Yixun):** "Accept, label R@1" — accept the exact acoustics and use exp_14's FLAC values for 25/50/75 %; label FLAC's R@1 as exp_14's evaluator (or omit it) in the data-efficiency table / plot.
- **Scope as recorded (narrow):** only the two E0a cells and only their R@1 keys, with the measured values pinned:

| cell | key | ours (exp_38 evaluator) | exp_14 record | items (of 6,337) |
|---|---|---|---|---|
| de_E0a_K8_s42 | RIR_to_GT_RIR_R@1 | 4.4343 | 4.3869 | +3 |
| de_E0a_K8_s42 | RIR_to_geom_R@1 | 3.519 | 3.5506 | −2 |
| de_E0a_K1_s42 | RIR_to_GT_RIR_R@1 | 4.4343 | 4.4185 | +1 |
| de_E0a_K1_s42 | RIR_to_geom_R@1 | 3.5348 | 3.5032 | +2 |

- **Every other key of E0a and every key of E0b–E0d stays type-strictly exact.** The code refuses a decision outside this registered shape (another gate / cell, a non-retrieval key, a reference value that is not the registered record's, ours == reference).
- **NOT covered by the decision as given (measured, same replay):** R@5 / R@10 also differ —
  K8: `RIR_to_GT_RIR_R@10` 22.8657 vs 22.8184 (+3), `RIR_to_geom_R@5` 12.6243 vs 12.719 (−6), `RIR_to_geom_R@10` 19.4572 vs 19.4729 (−1);
  K1: `RIR_to_GT_RIR_R@5` 14.4864 vs 14.5337 (−3), `RIR_to_geom_R@5` 12.2455 vs 12.1982 (+3), `RIR_to_geom_R@10` 19.4887 vs 19.394 (+6).
  Under the R@1-only decision a re-sealed E0 therefore FAILS on those six values (fail-closed). Widening the acceptance to all
  retrieval keys is Yixun's call; it is a one-file change (`accepted` gains these entries) and needs a new pin before the E0 re-run.
- **Downstream:** the FLAC 25/50/75 % rows (exp_14's dc_van records) keep exp_14's values; their retrieval columns carry `†` and the
  results JSON flags them `retrieval_evaluator: "exp_14"` (`exp38_collect.py`).
