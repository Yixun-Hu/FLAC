**Reviewer:** OpenAI Codex (GPT-5, codex-cli 0.147.0, `codex exec`, GitHub-connector read-only — local sandbox unavailable on della) · **Date:** 2026-08-21
**Round:** code round E (Phase-3 repro-eval kit + batched kit edits) · **Verdict:** REVISE

### BLOCKING

1. Same-cell exclusion ends before the jobs do. The wrapper lock serializes only concurrent submission transactions (`della_repro_eval_submit.sh:85-95`); a later invocation can resubmit cells that remain pending or running. A single explicit invocation also accepts duplicate cells (`della_repro_eval_submit.sh:103-109`). Duplicate processes open the same metrics and stream files with truncating writes (`eval_FLAC.py:1028-1039`) and then `cp -f` the same repository paths (`della_repro_eval.sbatch:229-236`), risking corruption or silent replacement. Reject duplicate arguments and add a cell-scoped lock held by the job through evaluation and copying, or an equivalently fail-closed active-cell gate. The 14 distinct cells must remain concurrent.

### NON-BLOCKING

1. The no-training-state gate is safe for this completed chain, but not for general reuse. Both submission and execution prove only that a checkpoint is nonempty (`della_repro_eval_submit.sh:126-135`; `della_repro_eval.sbatch:147-152`), not that its writer has finished. A future invocation during the final checkpoint save could read an in-flight artifact. Prefer a finalized-chain marker or a fail-closed active-training check modeled on `della_chain.sbatch:266-287`.

2. Submission preflights only the 67,500 checkpoint, even when endpoint cells are selected (`della_repro_eval_submit.sh:126-135`). The job-side gates prevent incorrect evaluation, and this run has all 27 checkpoints, but checking the selected 62,500/65,000 inputs before release would preserve the transaction’s fail-fast intent.

3. Repository documentation still says Phase-3 remains entirely on A100, while the approved kit runs 13 H200 cells plus one A100 spot-check (`plan_della_vanilla_repro.md:178-180`; `della_repro_eval.sbatch:46-50`). Record the later Yixun approval before analysis.

The cell table itself is correct: configs, seeds, counts, checkpoints, and eval names match Phase 3. Checkpoint-basename prefixing makes the 62,500/65,000/67,500 outputs distinct (`eval_FLAC.py:179-186`; `della_repro_eval.sbatch:196-198`). The A100 CLI clears partition/QOS and replaces GRES cleanly; 6 CPUs and 32 GB match the proven Phase-1 allocation. Distinct-cell logs, W&B/HF use, scratch files, and repository copies are concurrency-safe. Deferring `git add -f` and SHA256 generation to the analysis round is sound. The held→record→release transaction and absence of a training-only `PHASE1_PASS` interlock are also sound. All three batched edits are accepted: the two-hour eval limit retains ample measured margin, and the H200/8-CPU chain configuration completed all nine legs cleanly.

> Invocation note (Planner): GitHub-connector review at HEAD dc33112; batched round covering dc33112 + e2d77a2 + 624fc5a + 7b407d9 per SOP small-scripts clause.
