# Commits — exp_09 fa_finetune

- `d24e42b` scaffold + plan draft
- `01ea9ac` exp_09 plan Rev 2: all review findings applied (F-warm/F-reset probe-then-commit, fixed references, exp_08-faithful G1, 8/8 FULL bar, resume-validation probes); AWAITING Yixun approval
- `5e5785d` exp_09: plan APPROVED (Yixun) -> implementation round 1 started
- `b015a73` exp_09 round 1: f_arm_launch.sh (arm allow-list, RESUME/MAXSTEPS required, OPT_RESET via hook) + strip_optimizer_state hook (TDD, 10 tests; PL empty-not-absent finding w/ source refs; anchor-immutable)
- `bd619d8` exp_09 round 2: keep-entry/clear-state hook (r1 LR blocker fixed — empty-list = 96x understep, regression-pinned), variant identities Fw/Fr/V, CHECKPOINT_EVERY, lineage+SHA guards, 33-assertion guardtests; 58/58 green
- `7ca13cc` exp_09 round 3: Fr path-provenance gate + EXPECTED_STEP floor + guardtest sandboxing (39/39, independently re-verified)
- `d600ff3` exp_09: command log started; resume-validation probes launched
- `b701a6f` exp_09: variant pick = F-warm (online EDT+R@1 both better @88750); committed runs launching
- `6250ca2` exp_09 Fw screens: fine-tune-damage signature (all points worse than anchor, monotone degradation) — G2 FAIL likely; V + G1 pending for verdict
- `38990a7` exp_09: PROTOCOL ERROR — all fa evals ran cond_method=vanilla (mismatch artifact candidate); corrective fa_invariant eval block launched
- `e64c882` exp_09 VERDICT: exact C4 equivariance + Fw-95000 at released-Table-1 level (4 SUP/1 EQUIV/2 NONINF/1 OUT) — PARTIAL tier; damage narrative retired as protocol artifact
