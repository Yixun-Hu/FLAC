# Plan review request — exp_45 rf_data_curve, round 2 of at most 3

You are the Reviewer (OpenAI Codex). Read-only review. **Do not install, upgrade or modify any Python environment,
package, conda env or file. Do not run training, evaluation, sbatch, srun, pytest or any GPU job. Do not write
anywhere.** Cheap read-only commands (`git show`, `git diff`, `sha256sum`, `head`, `grep`, `python -c` that only reads)
are fine. Never run `find`, `du` or `ls -R` across `/n/fs`; read only the paths named here or in the plan.

Context: same as round 1 — read `rf_data_curve_codex_plan_r1_prompt.md` (this folder) for the file list, and your own
round-1 review `rf_data_curve_codex_plan_r1_review.md`. The object of review is the revised
`plan_rf_data_curve.md` (r1). Its §12 maps each of your 18 findings to a fix; §13 lists the changes.

Check, in one pass:
1. For each of the 18 round-1 findings: RESOLVED / PARTIAL / NOT RESOLVED, with a reason. Verify the new file:line
   claims in §3.2 and facts F13–F15 against the exp_38 kit in `/n/fs/gatrdp/codespace/exp-38-campaign`.
2. New defects the revision introduced (e.g. the supervisor moving `lightning_logs` into `receipts/` before the end
   audit — does anything else read `receipts/` and break; the shared storage floor; `EXTERNAL_CONFIGS` holding the
   68 reference files; the two-tree A6000 design and the records commit `Y` vs `cmd_run`'s HEAD check and the
   submitter's ancestry check; the bridge descriptor; the exp_38 offline adapter in `exp45_table.py`; the exp_45
   generator contract).
3. Anything still BLOCKING for a correct result. This is round 2; round 3 is the last, so classify carefully
   (BLOCKING / HIGH / MEDIUM / NIT), give file:line or plan section, scenario and fix.

End with a one-line verdict: APPROVE / APPROVE-WITH-CHANGES / REVISE.
