# Plan review request — exp_45 rf_data_curve, round 3 of 3 (THE LAST)

You are the Reviewer (OpenAI Codex). Read-only review. **Do not install, upgrade or modify any Python environment,
package, conda env or file. Do not run training, evaluation, sbatch, srun, pytest or any GPU job. Do not write
anywhere.** Cheap read-only commands are fine. Never run `find`, `du` or `ls -R` across `/n/fs`.

Context: as in rounds 1–2 (`rf_data_curve_codex_plan_r1_prompt.md` for the file list; your reviews
`rf_data_curve_codex_plan_r1_review.md`, `rf_data_curve_codex_plan_r2_review.md`). Object of review: the revised
`plan_rf_data_curve.md` (r2). §12 maps your round-2 findings r2-1 … r2-9 to fixes; §13 lists the changes.

This is the final round: there is no round 4. Anything still BLOCKING goes to the human (Yixun) as a decision, and
every other item is fixed without re-review. So, in one pass:
1. For each round-2 finding r2-1 … r2-9 (and the round-1 items you marked PARTIAL): RESOLVED / PARTIAL / NOT RESOLVED,
   with a reason, verified against the exp_38 kit in `/n/fs/gatrdp/codespace/exp-38-campaign` where it cites code.
2. New defects introduced by r2.
3. Report only what genuinely matters for a correct result or a safe run: BLOCKING / HIGH / MEDIUM / NIT, plan
   section or file:line, scenario, fix. Do not re-raise design choices that are explicitly put to Yixun (N1–N10)
   unless the plan describes them incorrectly.

End with a one-line verdict: APPROVE / APPROVE-WITH-CHANGES / REVISE.
