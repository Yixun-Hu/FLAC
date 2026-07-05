# Commits — exp_03_fa_invariant_cond

Base: `0dce4ce` (end of exp_02). Chronological; SOP-evolution commits made during the experiment included for lineage. Code-line counts per the <200 rule are in each round's notebook entry.

```
065cba5 SOP v2: reviewer reciprocity, lab notebook, validation ladder, parity audit, failure discipline
24319d9 SOP: code-review file must open with reviewer identity header
76f883d SOP: review filename carries the reviewer's name
9dabe7b SOP + announcement 02: mandatory test-driven development
fc703a0 exp_03: scaffold — Route 1 hard invariant conditioning (plan, query, notebook)
9e537c9 SOP/announcement 02: tests folder location is user-confirmed; FLAC uses src/tests/
a0fcc63 SOP: mandatory plan review before approval; reviews use reviewer's strongest model
ac9709a exp_03: plan review (Codex gpt-5.5 xhigh, REQUEST-CHANGES) + plan revision addressing all 8 findings
6b166c9 exp_03: hypothesis restructured as H1/H2/H3 (Yixun's correction)
031852d exp_03: tests for wrap_angle, cylindrical_pose_features, rotate pose_keys (RED)
a56e5e7 exp_03: implement wrap_angle + cylindrical_pose_features + rotate pose_keys (GREEN)
99640bb SOP: per-Coder-round code reviews with <marker> filenames
2828991 exp_03: tests for invariant_conditioning + MultiConditioner only_ids (RED)
0e00be0 exp_03: implement invariant_conditioning + MultiConditioner only_ids (GREEN)
fd2533f SOP: reviewer briefing rule (load worklog context before reviewing); exp_03 notebook cycle-3 entry
8180c9b exp_03: cycle-3 Codex review (REQUEST-CHANGES: finding-8 test hole)
7270899 SOP: a Coder round closes only after write->review->fix->re-verify completes
7091e83 exp_03: strengthen FakeGeometry + negative stale-depth test (review fix)
4a27fdd exp_03: notebook — round invcond closed (fix 7091e83 re-verified)
ca4f8b2 exp_03: cyl_pose Codex review (REQUEST-CHANGES: all-degenerate eps fallback)
70025d5 exp_03: all-degenerate eps fallback honors contract (review fix)
a645e80 exp_03: notebook — round cyl_pose closed (fix 70025d5 re-verified)
5fb9786 exp_03: tests for cond_method dispatch in all step methods (RED)
baf6902 exp_03: wire cond_method dispatch into wrapper + factory (GREEN)
5b95048 exp_03: ladder rungs a/c/d — H1 confirmed on real stack; degenerate sources real (11/6337 eval items)
61d4500 exp_03: dispatch review APPROVE — round closed, notebook updated
8e6164a exp_03: tests for eval output paths + comparator meta guard (RED)
337eec3 exp_03: eval_FLAC --cond-method fa_invariant + output-path/sidecar fixes (GREEN 11a)
1de5721 exp_03: comparator accepts prediction sidecar + meta guard (GREEN 11b)
167ea2d exp_03: rung b PASS — conditioning float-exact (5e-8), waveform floor is VAE-decoder-amplified noise; H1 criterion pre-registered
737249d exp_03: evalwire Codex review (REQUEST-CHANGES: meta-guard scope, method validation, wiring test)
37aa6bd exp_03: widen meta guard, validate cond_method in evaluate_model, wiring test (review fix)
ab9d0e2 exp_03: notebook — round evalwire closed (fix 37aa6bd re-verified)
6d94a45 exp_03: tests for finetune_cond config injection (RED)
bd03a5c exp_03: finetune_cond.py — non-destructive fine-tune driver (GREEN)
333acc7 exp_03: parity audit clean (4 intended deviations only) + rung e smoke evidence
b5fe113 exp_03: finetune Codex review (REQUEST-CHANGES: grad-clip drift, smoke checkpointing, recipe pins)
5cfffaf exp_03: grad-clip parity default, smoke checkpointing guarantee, recipe pins (review fix)
88f69b8 exp_03: notebook — round finetune closed (fix 5cfffaf re-verified); all TDD rounds complete
f5606df exp_03: integrative full review — GO-WITH-CONDITIONS (4 launch conditions)
992fe49 exp_03: launch conditions — cond-autocast control + load-integrity assertion (full-review fix)
665e101 exp_03: launch package — params, commands, pre-registered acceptance criteria (C3)
f472328 exp_03: --accumulate-grad-batches for shared-GPU capacity (launch adaptation)
0c86c3e exp_03: accum review APPROVE — launch docs updated to batch 4 x accum 2
17d353e exp_03: R0 done, probes closed, R1 launch logged
3bb7b3c exp_03: R1 gate FAIL (all primary metrics); EMA hypothesis + decision rule pre-registered
b6bddcc exp_03: corrected diagnostic (EMA hypothesis dead); amended iteration R1b = original-batch parity
e948d57 exp_03: living results doc — R0, R1 gate fail, diagnostics, R1b running
7bab5da exp_03: command.md — add missing R1b + diagnostic commands (doc gap)
29781a0 SOP: every launched run's command lands in _command.md at launch time
588f48c exp_03: R1b gate FAIL — registered stop; final results + analysis
```

*(this file's own commit: child of `588f48c`)*
