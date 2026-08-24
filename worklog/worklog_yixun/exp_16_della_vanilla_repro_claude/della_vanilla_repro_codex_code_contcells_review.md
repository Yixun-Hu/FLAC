**Reviewer:** OpenAI Codex (gpt-5, codex-cli 0.147.0, `codex exec`, GitHub-connector read-only — local sandbox unavailable on della) · **Date:** 2026-08-24
**Round:** code round G (continuous-arm eval cells) · **Verdict:** APPROVE

BLOCKING

NON-BLOCKING

- The transaction record still labels every submission with the chunked checkpoint. A `cont`-only or mixed submission will therefore record misleading checkpoint provenance, although execution uses the correct paths. Report the selected distinct checkpoints instead. `della_repro_eval_submit.sh:322`
- The usage preamble still describes `all` as “every registered cell (14)”; there are now 24 registered cells and `all` intentionally means the 14 chunked cells. `della_repro_eval_submit.sh:6-9`

Overall: all ten c-cells correctly select the unseen K8/K1 configs, seeds 42–46, count 6337, continuous 67,500-step checkpoint, and distinct `exp16_cont_` eval names (`della_repro_eval.sbatch:135-165`). `build_output_paths` uses the checkpoint directory plus checkpoint basename and eval name (`eval_FLAC.py:179-186`), matching the corrected artifact assertions and copy inputs (`della_repro_eval.sbatch:258-261`, `della_repro_eval.sbatch:295-300`). Remaining operational `CKD` uses are confined to the chunked checkpoints/endpoints; continuous preflight routes through `CKPT_CONT_67500` (`della_repro_eval_submit.sh:112-118`). Selector counts and meanings are correct, `ALL_CELLS` is used for explicit-name validation rather than selection, and preflight covers only selected cells while reporting distinct checkpoints (`della_repro_eval_submit.sh:87-98`, `della_repro_eval_submit.sh:159-176`, `della_repro_eval_submit.sh:193-218`). Same-cell exclusion, fail-closed real-submit scheduler checks, and write-free dry runs remain intact (`della_repro_eval_submit.sh:121-153`, `della_repro_eval_submit.sh:235-288`).

> Invocation note (Planner): GitHub-connector review at HEAD a4e9f7f. APPROVE; two cosmetic nits (record checkpoint label, usage text) batched for the next round.
